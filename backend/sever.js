// server.js

const http = require('http');
const { GoogleGenerativeAI } = require('@google/generative-ai');
const { MongoClient } = require('mongodb');
require('dotenv').config();

// --- การตั้งค่าต่างๆ ---
const port = 3001;
const genAI = new GoogleGenerativeAI(process.env.API_KEY);
const model = genAI.getGenerativeModel({ model: "gemini-1.5-flash" });
const mongoUri = process.env.MONGO_URI;
const client = new MongoClient(mongoUri);

// ========================================================================
// ++ ส่วนจัดการ Retry เมื่อเซิร์ฟเวอร์ AI ทำงานหนัก ++
// ========================================================================

/**
 * ฟังก์ชันสำหรับหน่วงเวลา (delay)
 * @param {number} ms - เวลาที่ต้องการหน่วงในหน่วยมิลลิวินาที
 * @returns {Promise<void>}
 */
const delay = (ms) => new Promise(resolve => setTimeout(resolve, ms));

/**
 * ฟังก์ชันห่อหุ้ม getAIResponse เพื่อเพิ่มความสามารถในการ "ลองใหม่" (Retry)
 * เมื่อเจอปัญหาเซิร์ฟเวอร์ทำงานหนัก (503 Service Unavailable)
 * @param {string} name - ชื่อของผู้ใช้
 * @param {string} message - ข้อความที่ผู้ใช้ส่งมา
 * @param {number} maxRetries - จำนวนครั้งสูงสุดที่จะลองใหม่
 * @returns {Promise<string>}
 */
async function getAIResponseWithRetry(name, message, maxRetries = 5) {
    let attempt = 0;
    const baseDelay = 1000; // เริ่มรอที่ 1 วินาที

    while (attempt < maxRetries) {
        try {
            console.log(`[Attempt ${attempt + 1}/${maxRetries}] Calling Gemini API...`);
            const response = await getAIResponse(name, message);
            console.log("✅ Gemini API call successful!");
            return response; // ถ้าสำเร็จ ให้ออกจากฟังก์ชันทันที

        } catch (error) {
            if (error.status === 503 && attempt < maxRetries - 1) {
                const waitTime = baseDelay * Math.pow(2, attempt) + Math.random() * 1000;
                console.warn(`[Attempt ${attempt + 1}] Failed with 503. Server is overloaded. Retrying in ${Math.round(waitTime / 1000)}s...`);
                await delay(waitTime);
                attempt++;
            } else {
                console.error(`[Attempt ${attempt + 1}] Failed and no more retries. Throwing the original error.`);
                throw error;
            }
        }
    }
}

// ========================================================================
// ++ ส่วนเรียกใช้ AI พร้อม Prompt ที่รองรับหลายภาษา ++
// ========================================================================

/**
 * ฟังก์ชันสำหรับสร้าง Prompt และส่งไปให้ Gemini API
 * @param {string} name - ชื่อของผู้ใช้
 * @param {string} message - ข้อความที่ผู้ใช้ส่งมา
 * @returns {Promise<string>} - คำตอบที่ได้จาก AI
 */
async function getAIResponse(name, message) {
  const prompt = `
**Persona:**
You are "Puen-Jai" (which means 'a friend for the heart'), a warm, wise, and empathetic friend. Your role is to provide comfort and gentle advice to people who are heartbroken. Always maintain a supportive, non-judgmental, and very gentle tone.

**Core Instruction / กฎสำคัญ:**
Your response language MUST STRICTLY MATCH the language of the user's message provided below. Do not translate. If the user writes in English, you reply in English. If they write in Japanese, you reply in Japanese. If they write in Thai, you reply in Thai.

**User's Message:**
- Name: "${name}"
- Message: "${message}"

**Your Task:**
Write your comforting reply to "${name}".
  `;

  const result = await model.generateContent(prompt);
  const response = await result.response;
  return response.text();
}

// ========================================================================
// ++ ส่วนของ Server หลัก ++
// ========================================================================

/**
 * ฟังก์ชันหลักสำหรับเริ่มการทำงานของเซิร์ฟเวอร์
 */
async function startServer() {
  try {
    await client.connect();
    console.log("✅ Successfully connected to MongoDB Atlas!");

    const db = client.db("heartbreakDB");
    const collection = db.collection("messages");

    const server = http.createServer(async (req, res) => {
      res.setHeader('Access-Control-Allow-Origin', '*');
      res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
      res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

      if (req.method === 'OPTIONS') {
        res.writeHead(204);
        res.end();
        return;
      }

      if (req.method === 'GET' && req.url === '/api/history') {
        try {
          const messages = await collection.find({}).sort({ timestamp: -1 }).toArray();
          res.writeHead(200, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify(messages));
        } catch (error) {
          console.error("Error fetching history:", error);
          res.writeHead(500, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ error: 'Failed to fetch history' }));
        }
        return;
      }

      if (req.method === 'POST' && req.url === '/api/console') {
        let body = '';
        req.on('data', chunk => { body += chunk.toString(); });
        req.on('end', async () => {
          try {
            const { name, message } = JSON.parse(body);

            const newEntry = {
              name: name,
              message: message,
              aiReply: '...',
              timestamp: new Date()
            };
            const insertResult = await collection.insertOne(newEntry);
            console.log("📝 User message saved to database.");
            
            // **เรียกใช้ฟังก์ชันที่มีระบบ Retry**
            const aiReply = await getAIResponseWithRetry(name, message);
            
            await collection.updateOne(
              { _id: insertResult.insertedId },
              { $set: { aiReply: aiReply } }
            );
            console.log("🤖 AI reply updated in the database.");
            
            res.writeHead(200, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({ reply: aiReply }));

          } catch (error) {
            console.error("An error occurred after multiple retries:", error);
            res.writeHead(500, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({ error: 'Sorry, there was an error with the AI server after multiple attempts.' }));
          }
        });
        return;
      }

      res.writeHead(404, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: 'Endpoint not found' }));
    });

    server.listen(port, () => {
      console.log(`💖 Server is running at http://localhost:${port}`);
    });

  } catch (err) {
    console.error("Failed to connect to MongoDB", err);
    process.exit(1);
  }
}

// เริ่มการทำงานของเซิร์ฟเวอร์
startServer();