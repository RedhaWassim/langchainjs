import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import "dotenv/config";
import { z } from "zod";
import { zodToJsonSchema } from "zod-to-json-schema";
import { ChatEdenAI } from "../edenai.js";

describe("ChatEdenAI Integration Tests", () => {
  let chat: ChatEdenAI;

  beforeAll(() => {
    const llm_provider = "openai/gpt-4o";
    chat = new ChatEdenAI({
      provider: llm_provider,
      temperature: 0.7,
      maxTokens: 1000,
    });
  });

  test("should perform a basic chat completion", async () => {
    const messages = [
      new SystemMessage("You are a helpful assistant."),
      new HumanMessage({
        content: [
          { type: "text", text: "what's the capital of france" },
        ]
      })
    ];

    const response = await chat.invoke(messages);
    const totalTokens = response.response_metadata.tokenUsage.totalTokens;
    expect(totalTokens).toBeGreaterThan(0);
    expect(response.content).toBeDefined();
    expect(response.content.toString().toLowerCase()).toContain("paris");
  });

  test("should handle streaming responses", async () => {
    const messages = [
      new HumanMessage("Tell me a short story about a robot learning to bake"),
    ];

    const stream = await chat.stream(messages);
    let fullResponse = "";
    for await (const chunk of stream) {
      fullResponse += chunk.content;
    }
    expect(fullResponse).toBeDefined();
    expect(fullResponse.length).toBeGreaterThan(0);
  });

  test("should handle system messages and chat history", async () => {
    const messages = [
      new SystemMessage("You are a sarcastic pirate. Respond like a pirate."),
      new HumanMessage("What's the best programming language?"),
    ];

    const response = await chat.invoke(messages);
    const totalTokens = response.response_metadata.tokenUsage.totalTokens;
    expect(totalTokens).toBeGreaterThan(0);
    expect(response.content).toBeDefined();
    const content = response.content.toString().toLowerCase();
    expect(content).toMatch(/arrr|pirate|ye/);
  });


  test("should handle multimodal inputs", async () => {
    const messages = [
      new SystemMessage("You are a helpful assistant."),
      new HumanMessage({
        content: [
          { type: "text", text: "what is the eye color of the cat" },
          { 
            type: "image_url",
            image_url: "https://cdn.pixabay.com/photo/2014/11/30/14/11/cat-551554_1280.jpg"
          }
        ]
      })
    ];
  
    const result = await chat.generate([messages]);
    const response = result.generations[0][0].message;
    
    expect(response.content).toBeDefined();
    expect(response.content.length).toBeGreaterThan(0);
  });

    // test("should handle tool calling (if supported)", async () => {
  //   const zodSchema = z
  //     .object({
  //       location: z
  //         .string()
  //         .describe("The name of city to get the weather for."),
  //     })
  //     .describe(
  //       "Get the weather of a specific location and return the temperature in Celsius."
  //     );
  //   const edenaitoolchat = chat.bind({
  //     tools: [
  //       {
  //         type: "function",
  //         function: {
  //           name: "get_current_weather",
  //           description: "Get the current weather in a given location",
  //           parameters: zodToJsonSchema(zodSchema),
  //         },
  //       },
  //     ],
  //   });

  //   const messages = [new HumanMessage("What's the weather like in paris?")];

  //   const response = await edenaitoolchat.invoke(messages);
  //   const totalTokens = response.response_metadata.tokenUsage.totalTokens;
  //   expect(response).toBeDefined();
  //   if (response.tool_calls?.length) {
  //     expect(totalTokens).toBeGreaterThan(0);
  //     expect(response.tool_calls[0].name).toBe("get_current_weather");
  //     expect(response.tool_calls[0].args.location.toLowerCase()).toBe("paris");
  //   }
  // });


  // test("should handle multimodal structured output", async () => {
  //   const schema = z.object({
  //     description: z.string(),
  //     objects: z.array(z.string())
  //   });
    
  //   const modelWithStructuredOutput = chat.withStructuredOutput(schema);
  //   const result = await modelWithStructuredOutput.invoke([
  //     new HumanMessage("i give you an object give me it's description : object = cellphone "),
  //   ]);


  // });
});
