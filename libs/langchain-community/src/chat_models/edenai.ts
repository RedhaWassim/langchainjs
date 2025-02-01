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
import { Runnable } from "@langchain/core/runnables";
import { BaseLanguageModelInput } from "@langchain/core/language_models/base";
import { CallbackManagerForLLMRun } from "@langchain/core/callbacks/manager";

interface EdenAIToolCall {
  name: string;
  arguments: string;
  id: string;
}

interface EdenAIChatRequest {
  providers: string;
  text: string;
  previous_history?: Array<{
    role: "user" | "assistant";
    message: string;
    tool_calls?: EdenAIToolCall[];
  }>;
  chatbot_global_action?: string;
  tool_results?: Array<{
    id: string;
    result: string;
  }>;
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
}

export interface ChatEdenAIParams extends BaseChatModelParams {
  provider?: string;
  model?: string;
  temperature?: number;
  maxTokens?: number;
  edenaiApiKey?: string;
}

function convertMessagesToEdenAIFormat(messages: BaseMessage[]): {
  text: string;
  previous_history: EdenAIChatRequest["previous_history"];
  chatbot_global_action?: string;
  tool_results?: EdenAIChatRequest["tool_results"];
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

  const previousHistory: EdenAIChatRequest["previous_history"] = [];
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

  apiUrl = "https://api.edenai.run/v2/text/chat";

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

    const response = await this.makeRequest(requestBody, false);
    const providerResponse = (response as EdenAIChatResponse)[this.provider];
    if (providerResponse.status === "fail") {
      throw new Error(providerResponse.error?.message || "EdenAI API error");
    }

    const assistantMessages = providerResponse.message.filter(
      (msg) => msg.role === "assistant"
    );

    const toolCalls = assistantMessages
      .filter((msg) => msg.tool_calls)
      .flatMap((msg) => msg.tool_calls);

    const message = new AIMessage({
      content: providerResponse.generated_text ?? "",
      tool_calls: toolCalls
        .filter((call): call is EdenAIToolCall => call !== null)
        .map((call) => ({
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
    const formattedMessages = convertMessagesToEdenAIFormat(messages);

    const requestBody: EdenAIChatRequest = {
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

  override bindTools(
    tools: Array<{
      type: "function";
      function: {
        name: string;
        description: string;
        parameters: Record<string, unknown>;
      };
    }>,
    kwargs?: Partial<ChatEdenAICallOptions>
  ): Runnable<BaseLanguageModelInput, AIMessageChunk, ChatEdenAICallOptions> {
    return this.bind({
      tools: tools.map((tool) => convertToOpenAITool(tool)),
      ...kwargs,
    } as ChatEdenAICallOptions);
  }

  private async makeRequest(body: EdenAIChatRequest, stream: boolean) {
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
      body: JSON.stringify({ ...body, stream, show_original_response: true }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(
        `EdenAI API Error: ${error.error?.message || response.statusText}`
      );
    }

    return stream ? response : ((await response.json()) as EdenAIChatResponse);
  }
}
