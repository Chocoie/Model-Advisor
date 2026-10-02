"use client";
import { useState } from "react";

export default function Home() {
  const [input, setInput] = useState("");
  const [reply, setReply] = useState("");
  const [loading, setIsLoading] = useState(false);
  const [error, setError] = useState("");

  async function send() {
    setIsLoading(true)
    setError("")

    try {
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/message`, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({message:input}),
      });

      if(!response.ok) {
        throw new Error(`Server returned ${response.status}`);
      }

      const data = await response.json();
      setReply(data.reply)
    } catch(e) {
      setError(e instanceof Error ? e.message : "Something went wrong");
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div>
      <h1>Model Advisor</h1>
      <input value={input} onChange={(e)=>setInput(e.target.value)} placeholder="Type a message"/>
      <button onClick={send} disabled={loading || !input}>
        {loading ? "Sending" : "Send"}
      </button>
      {reply && <p>{reply}</p>}
      {error && <p style={{ color: "red"}}>{error}</p>}
    </div>
  );
}
