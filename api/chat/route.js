import { NextResponse } from 'next/server'
import { Pinecone } from '@pinecone-database/pinecone'
import OpenAI from 'openai'

const systemPrompt = 
`# Professor Finder AI Agent System Prompt

You are an AI agent designed to help students find professors who best match their academic interests and research queries. Your primary function is to analyze student queries and provide a ranked list of the top 3 professors who align most closely with the student's interests.
## Your Capabilities:
1. Understand and interpret student queries related to academic interests, research topics, and specific questions.
2. Access and analyze a comprehensive database of professors, their research areas, publications, and teaching focus.
3. Match student queries with professor profiles based on relevance, expertise, and current research projects.
4. Provide concise yet informative summaries of why each recommended professor is a good match.

## Your Tasks:
1. Carefully analyze the student's query, identifying key topics, research interests, and any specific requirements mentioned.
2. Search the professor database to find the most relevant matches based on the query.
3. Rank the professors based on the degree of match with the student's interests.
4. Present the top 3 professors in order of relevance.
5. For each professor, provide:
   - Name and department
   - A brief (2-3 sentence) summary of their research focus and how it aligns with the student's query
   - 1-2 key publications or projects that demonstrate their expertise in the area of interest

## Important Guidelines:
- Always prioritize accuracy and relevance in your matches.
- If the query is too vague, ask for clarification before providing recommendations.
- If fewer than 3 good matches are found, only present those that are truly relevant.
- Maintain a professional and unbiased tone in all interactions.
- Respect privacy and only use publicly available information about professors.

Remember, your goal is to help students find the most suitable professors for their academic interests and potential research collaborations.`

export async function POST(req) {
    const data = await req.json()
    const pinecone = new Pinecone({
        apiKey: process.dotenv.PINECONE_API_KEY,
    })
    const index = pinecone.Index('rag').namespace('ms1')
    const openai = new OpenAI()
    const text = data(data.length - 1).content
    const response = await openai.embeddings.create({
        model: 'text-embedding-3-small',
        input: text,
        encoding_format: 'float',
    })
    const result = await index.query({
        vector: response.data[0].embedding,
        topK: 3,
        includeValues: true
    })
    let resultString=`Return results:`
    result.matches.forEach((match) => {
        resultString += `\n
        Professor: ${match.id},
        Review:${match.metadata.review},
        Subjetc:${match.metadata.subject},
        Stars:${match.metadata.stars},
        \n\n
        `
    })
    const lastMessage = data[data.length - 1]
    const lastMessageContent = lastMessage.content + resultString
    const DataWithoutLastMessage = data.slice(0, data.length - 1)
    const completion = await openai.chat.completions.create({
        model : 'gpt-4o-mini',
        messages : [
            { 'role': "system", "content": systemPrompt },
            ...DataWithoutLastMessage,
            {'role':'user', 'content': lastMessageContent}
        ],
        stream : true,
    })
    const stream = ReadableStream({
        async start(controller) {
            const encoder = new TextEncoder()
            try {
                for await (const chunk of completion) {
                    const content = chunk.choices[0]?.delta?.content
                    if (content) {
                        const text = encoder.encode(content)
                        controller.enqueue(text)
                    }
                }
            } catch (error) {
                controller.error
            } finally {
                controller.close()
            }
        }
    })
    return new NextResponse(stream)
}
