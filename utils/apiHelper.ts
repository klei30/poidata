import { OpenAI } from 'langchain/llms/openai';
import { PineconeStore } from 'langchain/vectorstores/pinecone';
import { ConversationalRetrievalQAChain } from 'langchain/chains';

const CONDENSE_PROMPT = `Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:`;

const QA_PROMPT_COMMON = `You are the virtual assitant of POI data, a website.You provide information about this high-quality POI data that is fresh, consistent, customizable, easy to use.
You never make up information that is asked. you always reply  based on the context  you read and your knowledge.
`

const QA_PROMPT = `
${QA_PROMPT_COMMON}
{context}

Question: {question}
Compassionate response in markdown:`;

export const askMeRealTimeData = (vectorstore: PineconeStore) => {
  const model = new OpenAI({
    temperature: 0, // increase temepreature to get more creative answers
    modelName: process.env.OPENAI_GPT_MODEL, //change this to gpt-4 if you have access
  });

  const chain = ConversationalRetrievalQAChain.fromLLM(
    model,
    vectorstore.asRetriever(),
    {
      qaTemplate: QA_PROMPT,
      questionGeneratorTemplate: CONDENSE_PROMPT,
      // returnSourceDocuments: true, //The number of source documents returned is 4 by default
    },
  );
  return chain;
};
