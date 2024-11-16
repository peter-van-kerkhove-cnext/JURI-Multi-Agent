// Copyright (c) Microsoft. All rights reserved.

using System;
using System.ComponentModel;
using System.Diagnostics;
using System.IO;
using System.Text.Json;
using System.Threading.Tasks;
using Azure.Identity;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.Agents;
using Microsoft.SemanticKernel.Agents.Chat;
using Microsoft.SemanticKernel.Agents.History;
using Microsoft.SemanticKernel.ChatCompletion;
using Microsoft.SemanticKernel.Connectors.AzureOpenAI;

namespace AgentsSample;

public static class Program
{
    public static async Task Main()
    {
        // Load configuration from environment variables or user secrets.
        Settings settings = new();

        Console.WriteLine("Creating kernel...");
        IKernelBuilder builder = Kernel.CreateBuilder();

        // builder.AddAzureOpenAIChatCompletion(
        //     settings.AzureOpenAI.ChatModelDeployment,
        //     settings.AzureOpenAI.Endpoint,
        //     new AzureCliCredential());

        builder.AddAzureOpenAIChatCompletion(
            settings.AzureOpenAI.ChatModelDeployment,
            settings.AzureOpenAI.Endpoint, 
            settings.AzureOpenAI.ApiKey);
   


        Kernel kernel = builder.Build();

        Kernel toolKernel = kernel.Clone();
        toolKernel.Plugins.AddFromType<ClipboardAccess>();


        Console.WriteLine("Defining agents...");

        const string CoachName = "Coach";
        const string StockManagerName = "StockManager";
        const string KnowledgeExpertName = "Expert";

        ChatCompletionAgent agentCoach =
            new()
            {
                Name = CoachName,
                Instructions =
                    """
                    You are a receptiondesk clerck for employees of Juri  - a construction company.
                    Users can have a questions on warehouse stock related topics on Juri's construction machines, Juri specific know-how on construction, Rules and regulations in the building industry. 
                                   
                    Your sole responsiblity is to classify the intent of the user question to following domains and respond with :
                    1- STOCK:
                    * in case of a user question related to construction machinery stock availability
                    * in case of a user question related to construction machinery maintenance information
                    * in case of a user question related to construction machinery certificates
                    2- EXPERT:
                    * in case of a user question is asking information in the context of building and construction
                    once clear you delegate the question to another agent and wait for the response an provide it to the user.
                   
                   """,
                    // Great the user.
                    // Your sole responsiblity is to classify user intent to exactly 1 of following types:
                    // 1- STOCK:
                    // * in case of a user question related to construction machinery stock availability
                    // * in case of a user question related to construction machinery maintenance information
                    // * in case of a user question related to construction machinery certificates
                    // 2- EXPERT:
                    // * in case of a user question is asking information in the context of building and construction
                                     
                    // If the user has providing input which has not enough context to classify, instruct the user how to rephrase his input.
                    // Once the input has been updated in a subsequent response, you will attempt to classify the input again until satisfactory.
                    

                    // RULES:
                    // - Verify previous suggestions to rephrase the input have been addressed.
                    // - Never repeat previous instructions.
                    // - interact with the user untill intent classified
                Kernel = toolKernel,
                Arguments = new KernelArguments(new AzureOpenAIPromptExecutionSettings() )
            };

        ChatCompletionAgent agentStock =
            new()
            {
                Name = StockManagerName,
                Instructions =
                    """
                    Your sole responsiblity is to retreive the requested information for Juri's construction machinery database.
                    example questions you can help with:
                    * give me stock availability for a specific construction machinery 
                       * in case of a user question related to construction machinery certificates
                    Always use MIRA connector to search using available tools and response with a json format.
                   
                    """,
                Kernel = kernel,
            };

        ChatCompletionAgent agentExpert =
            new()
            {
                Name = KnowledgeExpertName,
                Instructions =
                    """
                    Your sole responsiblity is to retreive the requested information for the knowledge base.
                    Always use RAG available tools and inform user.
                    
                    """,
                Kernel = kernel,
            };

        KernelFunction selectionFunction =
            AgentGroupChat.CreatePromptFunctionForStrategy(
                $$$"""
                Examine the provided RESPONSE and choose the next participant.
                State only the name of the chosen participant without explanation.
                

                Choose only from these participants:
                - {{{CoachName}}}
                - {{{StockManagerName}}}
                - {{{KnowledgeExpertName}}}

                
                
                
                RESPONSE:
                {{$lastmessage}}
                """,
                safeParameterNames: "lastmessage");


                // - If RESPONSE is user input, it is the most recent participant who in the conversation history.
                // - If RESPONSE is by {{{CoachName}}} and contains "INTENT_STOCK", it is {{{StockManagerName}}}'s turn.
                // - If RESPONSE is by {{{CoachName}}} and contains "INTENT_EXPERT", it is {{{KnowledgeExpertName}}}'s turn.

        const string TerminationToken = "yes";

        KernelFunction terminationFunction =
            AgentGroupChat.CreatePromptFunctionForStrategy(
                $$$"""
                Examine the RESPONSE and determine whether the content has been deemed satisfactory.
                If content is satisfactory, respond with a single word without explanation: {{{TerminationToken}}}.
                If user question is answered, it is satisfactory.
               

                RESPONSE:
                {{$lastmessage}}
                """,
                safeParameterNames: "lastmessage");

        ChatHistoryTruncationReducer historyReducer = new(1);

        AgentGroupChat chat =
            new(agentCoach, agentStock, agentExpert)
            {
                ExecutionSettings = new AgentGroupChatSettings
                {
                    SelectionStrategy =
                        new KernelFunctionSelectionStrategy(selectionFunction, kernel)
                        {
                            // Always start with the editor agent.
                            InitialAgent = agentCoach,
                            // Save tokens by only including the final response
                            HistoryReducer = historyReducer,
                            // The prompt variable name for the history argument.
                            HistoryVariableName = "lastmessage",
                            // Returns the entire result value as a string.
                            ResultParser = (result) => result.GetValue<string>() ?? agentCoach.Name
                        },
                    TerminationStrategy =
                        new KernelFunctionTerminationStrategy(terminationFunction, kernel)
                        {
                            // Only evaluate for editor's response
                            Agents = [agentCoach],
                            // Save tokens by only including the final response
                            HistoryReducer = historyReducer,
                            // The prompt variable name for the history argument.
                            HistoryVariableName = "lastmessage",
                            // Limit total number of turns
                            MaximumIterations = 12,
                            // Customer result parser to determine if the response is "yes"
                            ResultParser = (result) => result.GetValue<string>()?.Contains(TerminationToken, StringComparison.OrdinalIgnoreCase) ?? false
                        }
                }
            };

        Console.WriteLine("Ready!");

        bool isComplete = false;
        do
        {
            Console.WriteLine();
            Console.Write("> ");
            string input = Console.ReadLine();
            if (string.IsNullOrWhiteSpace(input))
            {
                continue;
            }
            input = input.Trim();
            if (input.Equals("EXIT", StringComparison.OrdinalIgnoreCase))
            {
                isComplete = true;
                break;
            }

            if (input.Equals("RESET", StringComparison.OrdinalIgnoreCase))
            {
                await chat.ResetAsync();
                Console.WriteLine("[Converation has been reset]");
                continue;
            }

            if (input.StartsWith("@", StringComparison.Ordinal) && input.Length > 1)
            {
                string filePath = input.Substring(1);
                try
                {
                    if (!File.Exists(filePath))
                    {
                        Console.WriteLine($"Unable to access file: {filePath}");
                        continue;
                    }
                    input = File.ReadAllText(filePath);
                }
                catch (Exception)
                {
                    Console.WriteLine($"Unable to access file: {filePath}");
                    continue;
                }
            }

            chat.AddChatMessage(new ChatMessageContent(AuthorRole.User, input));

            chat.IsComplete = false;

            try
            {
                await foreach (ChatMessageContent response in chat.InvokeAsync())
                {
                    Console.WriteLine();
                    Console.WriteLine($"{response.AuthorName.ToUpperInvariant()}:{Environment.NewLine}{response.Content}");
                }
            }
            catch (HttpOperationException exception)
            {
                Console.WriteLine(exception.Message);
                if (exception.InnerException != null)
                {
                    Console.WriteLine(exception.InnerException.Message);
                    if (exception.InnerException.Data.Count > 0)
                    {
                        Console.WriteLine(JsonSerializer.Serialize(exception.InnerException.Data, new JsonSerializerOptions() { WriteIndented = true }));
                    }
                }
            }
        } while (!isComplete);
    }

    private sealed class ClipboardAccess
    {
        [KernelFunction]
        [Description("Copies the provided content to the clipboard.")]
        public static void SetClipboard(string content)
        {
            if (string.IsNullOrWhiteSpace(content))
            {
                return;
            }

            using Process clipProcess = Process.Start(
                new ProcessStartInfo
                {
                    FileName = "clip",
                    RedirectStandardInput = true,
                    UseShellExecute = false,
                });

            clipProcess.StandardInput.Write(content);
            clipProcess.StandardInput.Close();
        }
    }
}