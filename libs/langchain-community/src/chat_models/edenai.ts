import {
  BaseChatModel,
  type BaseChatModelParams,
  type BaseChatModelCallOptions,
} from "@langchain/core/language_models/chat_models";
import {
  AIMessage,
  type BaseMessage,
  HumanMessage,
  SystemMessage,
  ToolMessage,
  AIMessageChunk,
} from "@langchain/core/messages";
import { ChatResult, ChatGenerationChunk } from "@langchain/core/outputs";
import { getEnvironmentVariable } from "@langchain/core/utils/env";
import { convertToOpenAITool } from "@langchain/core/utils/function_calling";
import { CallbackManagerForLLMRun } from "@langchain/core/callbacks/manager";

// import { Runnable } from "@langchain/core/runnables";
// import { BaseLanguageModelInput } from "@langchain/core/language_models/base";
// import zodToJsonSchema from "zod-to-json-schema";
// import { JsonOutputParser } from "@langchain/core/output_parsers";
// import { z } from "zod";

interface EdenAIToolCall {
  name: string;
  arguments: string;
  id: string;
}

interface EdenAIMediaContent {
  type: "text" | "media_url" | "media_base64";
  content: {
    text?: string;
    media_url?: string;
    media_type?: string;
    media_base64?: string;
  };
}

interface EdenAIChatRequest {
  providers: string;
  messages: Array<{
    role: "user" | "assistant";
    content: EdenAIMediaContent[];
    tool_calls?: EdenAIToolCall[];
  }>;
  chatbot_global_action?: string;
  tool_results?: Array<{ id: string; result: string }>;
  temperature?: number;
  max_tokens?: number;
  fallback_providers?: string;
  settings?: Record<string, string>;
  available_tools?: Array<{
    name: string;
    description: string;
    parameters: Record<string, unknown>;
  }>;
  show_original_response?: boolean;
  response_format?: {
    type: "json_schema";
    json_schema?: Record<string, unknown>;
    strict?: boolean;
  };
}

interface EdenAIChatStreamRequest {
  providers: string;
  text: string
  chatbot_global_action?: string;
  tool_results?: Array<{ id: string; result: string }>;
  temperature?: number;
  max_tokens?: number;
  fallback_providers?: string;
  settings?: Record<string, string>;
  available_tools?: Array<{
    name: string;
    description: string;
    parameters: Record<string, unknown>;
  }>;
  show_original_response?: boolean;
  response_format?: {
    type: "json_schema";
    json_schema?: Record<string, unknown>;
    strict?: boolean;
  };
  previous_history?: Array<{
    role: "user" | "assistant";
    message: string;
    tool_calls?: EdenAIToolCall[];
  }>;
}

interface EdenAIToolCall {
  id: string;
  name: string;
  arguments: string;
}

interface EdenAIToolMessage {
  role: "user" | "assistant";
  message: string | null;
  name: string;
  description: string;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  parameters: Record<string, any>;
  tool_calls: Array<EdenAIToolCall> | null;
}

interface EdenAIChatResponse {
  [provider: string]: {
    original_response: any;
    generated_text: string | null;
    status: "success" | "fail";
    message: Array<EdenAIToolMessage>;
    error?: {
      message: string;
    };
  };
}
interface MultimodalEdenAIChatResponse {
  [provider: string]: {
    original_response: any;
    generated_text: string | null;
    status: "success" | "fail";
    messages: Array<EdenAIToolMessage>;
    error?: {
      message: string;
    };
  };
}

export interface ChatEdenAICallOptions extends BaseChatModelCallOptions {
  fallback_providers?: string;
  tools?: Array<{
    type: "function";
    function: {
      name: string;
      description: string;
      parameters: Record<string, unknown>;
    };
  }>;
  response_format?: {
    type: "json_schema";
    json_schema?: Record<string, unknown>;
    strict?: boolean;
  };
}

export interface ChatEdenAIParams extends BaseChatModelParams {
  provider?: string;
  model?: string;
  temperature?: number;
  maxTokens?: number;
  edenaiApiKey?: string;
}

function convertMessagesToEdenAIFormat(messages: BaseMessage[]): {
  messages: EdenAIChatRequest["messages"];
  chatbot_global_action?: string;
  tool_results?: EdenAIChatRequest["tool_results"];
} {
  const toolResults: Array<{ id: string; result: string }> = [];
  const convertedMessages: EdenAIChatRequest["messages"] = [];
  let systemMessage: string | undefined;

  for (const message of messages) {
    // eslint-disable-next-line no-instanceof/no-instanceof
    if (message instanceof ToolMessage) {
      toolResults.push({
        id: message.tool_call_id,
        result: Array.isArray(message.content) 
          ? message.content.join(", ") 
          : message.content,
      });
    // eslint-disable-next-line no-instanceof/no-instanceof
    } else if (message instanceof SystemMessage) {
      systemMessage = Array.isArray(message.content)
        ? message.content.join(", ")
        : message.content;
    } else {
      const role = message._getType() === "human" ? "user" : "assistant";
      const content: EdenAIMediaContent[] = [];

      if (typeof message.content === "string") {
        content.push({
          type: "text",
          content: { text: message.content }
        });
      } else if (Array.isArray(message.content)) {
        for (const part of message.content) {
          if (part.type === "text") {
            content.push({
              type: "text",
              content: { text: part.text }
            });
             } else if (part.type === "media_url") {
            const url = part.image_url;
            content.push({
              type: "media_url",
              content: {
                media_url: url,
                media_type: "image/jpeg"
              }
             })
          } else if (part.type === "media_base64") {
            content.push({
              type: "media_base64",
              content: {
                media_base64: part.media_base64,
                media_type: part.media_type
              }
            });
          }
        }
      }

      convertedMessages.push({
        role,
        content,
        tool_calls: "tool_calls" in message && Array.isArray(message.tool_calls) ? message.tool_calls.map(tc => ({
          id: tc.id,
          name: tc.name,
          arguments: JSON.stringify(tc.args)
        })) : undefined
      });
    }
  }


  return {
    messages: convertedMessages,
    chatbot_global_action: systemMessage,
    tool_results: toolResults.length > 0 ? toolResults : undefined
  };
}



function convertMessagesToEdenAIStreamFormat(messages: BaseMessage[]): {
  text: string;
  previous_history: EdenAIChatStreamRequest["previous_history"];
  chatbot_global_action?: string;
  tool_results?: EdenAIChatStreamRequest["tool_results"];
} {
  const toolResults: Array<{ id: string; result: string }> = [];
  const filteredMessages: BaseMessage[] = [];
  let systemMessage: string | undefined;

  for (let i = messages.length - 1; i >= 0; i -= 1) {
    const message = messages[i];
    // eslint-disable-next-line no-instanceof/no-instanceof
    if (message instanceof ToolMessage) {
      toolResults.unshift({
        id: message.tool_call_id,
        result: Array.isArray(message.content)
          ? message.content.join(", ")
          : message.content,
      });
    } else {
      filteredMessages.unshift(message);
    }
  }

  const previousHistory: EdenAIChatStreamRequest["previous_history"] = [];
  let lastHumanMessage = "";

  for (const message of filteredMessages) {
    // eslint-disable-next-line no-instanceof/no-instanceof
    if (message instanceof SystemMessage) {
      systemMessage = Array.isArray(message.content)
        ? message.content.join(", ")
        : message.content;
      // eslint-disable-next-line no-instanceof/no-instanceof
    } else if (message instanceof HumanMessage) {
      lastHumanMessage = Array.isArray(message.content)
        ? message.content.join(", ")
        : message.content;
      // eslint-disable-next-line no-instanceof/no-instanceof
    }
  }

  return {
    text: lastHumanMessage,
    previous_history: previousHistory,
    chatbot_global_action: systemMessage,
    tool_results: toolResults.length > 0 ? toolResults : undefined,
  };
}

export class ChatEdenAI extends BaseChatModel<ChatEdenAICallOptions> {
  static lc_name() {
    return "ChatEdenAI";
  }

  provider: string;

  model?: string;

  temperature: number;

  maxTokens?: number;

  edenaiApiKey: string;

  apiUrl = "https://api.edenai.run/v2/multimodal/chat";

  constructor(params: ChatEdenAIParams = {}) {
    super(params);

    this.provider = params.provider || "openai";
    this.model = params.model;
    this.temperature = params.temperature ?? 0.7;
    this.maxTokens = params.maxTokens;
    this.edenaiApiKey =
      params.edenaiApiKey || getEnvironmentVariable("EDENAI_API_KEY") || "";
    if (!this.edenaiApiKey) {
      throw new Error("EdenAI API key is required");
    }
  }

  _llmType() {
    return "edenai-chat";
  }

  async _generate(
    messages: BaseMessage[],
    options: this["ParsedCallOptions"],
    runManager?: CallbackManagerForLLMRun
  ): Promise<ChatResult> {
    const formattedMessages = convertMessagesToEdenAIFormat(messages);
    
    const requestBody: EdenAIChatRequest = {
      providers: this.provider,
      ...formattedMessages,
      temperature: this.temperature,
      max_tokens: this.maxTokens,
      fallback_providers: options.fallback_providers,
      settings: this.model ? { [this.provider]: this.model } : undefined,
      available_tools: options.tools?.map(tool => ({
        name: tool.function.name,
        description: tool.function.description || "",
        parameters: tool.function.parameters
      })),
      show_original_response: true,
      response_format: options.response_format
    };

    const response = await this.makeRequest(requestBody, false);
    const providerResponse = (response as MultimodalEdenAIChatResponse)[this.provider];
    if (providerResponse.status === "fail") {
      throw new Error(providerResponse.error?.message || "EdenAI API error");
    }

    const assistantMessage = providerResponse.messages
    .filter((msg) => msg.role === "assistant")
    .at(-1); 

    if (!assistantMessage) {
      throw new Error("No assistant message found in response");
    }

    const toolCalls = assistantMessage.tool_calls || [];

    const message = new AIMessage({
      content: providerResponse.generated_text ?? "",
      tool_calls: toolCalls.map((call) => ({
        name: call.name,
        args: JSON.parse(call.arguments),
        id: call.id,
      })),
    });

    const tokenUsage = providerResponse.original_response?.usage;

    return {
      generations: [
        {
          message,
          text: providerResponse.generated_text ?? "",
        },
      ],
      llmOutput: {
        ...response,
        tokenUsage: tokenUsage
          ? {
              promptTokens: tokenUsage.prompt_tokens ?? 0,
              completionTokens: tokenUsage.completion_tokens ?? 0,
              totalTokens: tokenUsage.total_tokens ?? 0,
            }
          : { promptTokens: 0, completionTokens: 0, totalTokens: 0 },
      },
    };
  }

  async *_streamResponseChunks(
    messages: BaseMessage[],
    options: this["ParsedCallOptions"],
    runManager?: CallbackManagerForLLMRun
  ): AsyncGenerator<ChatGenerationChunk> {
    const formattedMessages = convertMessagesToEdenAIStreamFormat(messages);

    const requestBody: EdenAIChatStreamRequest = {
      providers: this.provider,
      ...formattedMessages,
      temperature: this.temperature,
      max_tokens: this.maxTokens,
      fallback_providers: options.fallback_providers,
      settings: this.model ? { [this.provider]: this.model } : undefined,
      available_tools: options.tools?.map((tool) => {
        const convertedTool = convertToOpenAITool(tool);
        return {
          name: convertedTool.function.name,
          description: convertedTool.function.description || "",
          parameters: convertedTool.function.parameters,
        };
      }),
      show_original_response: true,
    };

    const response = await this.makeRequest(requestBody, true);

    if (!response.body) {
      throw new Error("No response body received");
    }

    const reader = (response.body as ReadableStream<Uint8Array>).getReader();
    const decoder = new TextDecoder();

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value);
      const cleanedChunk = chunk.trim();
      let parsedChunk;
      try {
        parsedChunk = JSON.parse(cleanedChunk);
      } catch (error) {
        continue;
      }
      const assistantMessages = parsedChunk.message?.filter(
        (msg: any) => msg.role === "assistant"
      );

      const toolCalls = assistantMessages
        ?.filter((msg: any) => msg.tool_calls)
        .flatMap((msg: any) => msg.tool_calls);

      const chunkObj = new ChatGenerationChunk({
        message: new AIMessageChunk({
          content: parsedChunk.text,
          tool_call_chunks: toolCalls?.map((tc: any) => ({
            name: tc.name,
            args: tc.arguments,
            id: tc.id,
          })),
        }),
        text: parsedChunk.text,
      });

      yield chunkObj;
      await runManager?.handleLLMNewToken(parsedChunk.text);
    }
  }


  private async makeRequest(body: EdenAIChatRequest | EdenAIChatStreamRequest, stream: boolean) {
    const headers = {
      Authorization: `Bearer ${this.edenaiApiKey}`,
      "Content-Type": "application/json",
    };

    const apiUrl = stream
      ? "https://api.edenai.run/v2/text/chat/stream"
      : this.apiUrl;
    const response = await fetch(apiUrl, {
      method: "POST",
      headers,
      body: JSON.stringify({ ...body, show_original_response: true }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(
        `EdenAI API Error: ${error.error?.message || response.statusText}`
      );
    }

    return stream ? response : ((await response.json()) as MultimodalEdenAIChatResponse);
  }

  // override bindTools(
  //   tools: Array<{
  //     type: "function";
  //     function: {
  //       name: string;
  //       description: string;
  //       parameters: Record<string, unknown>;
  //     };
  //   }>,
  //   kwargs?: Partial<ChatEdenAICallOptions>
  // ): Runnable<BaseLanguageModelInput, AIMessageChunk, ChatEdenAICallOptions> {
  //   return this.bind({
  //     tools: tools.map((tool) => convertToOpenAITool(tool)),
  //     ...kwargs,
  //   } as ChatEdenAICallOptions);
  // }


  // withStructuredOutput<RunOutput extends Record<string, any> = Record<string, any>>(
  //   schema: z.ZodType<RunOutput> | Record<string, any>,
  //   config?: { name?: string; strict?: boolean }
  // ): Runnable<BaseLanguageModelInput, RunOutput> {
  //   // eslint-disable-next-line no-instanceof/no-instanceof
  //   const jsonSchema = schema instanceof z.ZodType 
  //     ? zodToJsonSchema(schema)
  //     : schema;

  //   return this.bind({
  //     response_format: {
  //       type: "json_schema",
  //       json_schema: jsonSchema,
  //       strict: config?.strict
  //     }
  //   }).pipe(new JsonOutputParser());
  // }
}

