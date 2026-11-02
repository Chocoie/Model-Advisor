"use client";
import { useEffect, useState, useRef } from "react";

type CurrentUser = {
    email: string;
};

type Conversation = {
    conversation_id: string;
    title: string;
}

type ChatMessage = {
    role: "user" | "assistant";
    content: string;
};

const API_URL = "http://localhost:8000";

export default function Dialog() {
    const [user, setUser] = useState<CurrentUser | null>(null);
    const [error, setError] = useState("");
    const [conversations, setConversations] = useState<Conversation[]>([]);
    const [currConversation, setCurrConversation] = useState<string | null>(null);
    const [messages, setMessages] = useState<ChatMessage[]>([]);
    const [sending, setSending] = useState(false);

    useEffect(() => {
        async function loadUser() {
            const token = localStorage.getItem("_token");
            if (!token) {
                setError("No login session found. Please log in again.");
                return;
            }

            try {
                const response = await fetch("http://localhost:8000/auth/me", {
                    headers: {
                        Accept: "application/json",
                        Authorization: `Bearer ${token}`,
                    },
                });

                if (!response.ok) {
                    setError(response.status === 401
                        ? "Your login session has expired. Please log in again."
                        : "Could not load your user information.");
                    return;
                }

                setUser(await response.json() as CurrentUser);
                await loadConversations();
            } catch {
                setError("Could not connect to the server. Please try again.");
            }
        }

        void loadUser();
    }, []);

    // load the list of the user's conversations for the sidebar
    async function loadConversations() {
        try {
            const token = localStorage.getItem("_token");
            const response = await fetch(`${API_URL}/conversations`, {
                headers: { Authorization: `Bearer ${token}` },
            });
            if (!response.ok) return;
            const data = await response.json();
            setConversations(data.conversations ?? []);
        } catch {
            // sidebar just stays empty if this fails
        }
    }

    // open a previous conversation and load its messages
    async function openConversation(conversationId: string) {
        setCurrConversation(conversationId);
        setMessages([]);
        try {
            const token = localStorage.getItem("_token");
            const response = await fetch(`${API_URL}/conversations/${conversationId}`, {
                headers: { Authorization: `Bearer ${token}` },
            });
            if (!response.ok) return;
            const data = await response.json();
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            setMessages((data.messages ?? []).map((m: any) => ({
                role: m.role === "user" ? "user" : "assistant",
                content: m.content ?? m.message ?? "",
            })));
        } catch {
            setMessages([{ role: "assistant", content: "Error: could not load this conversation" }]);
        }
    }

    // start a fresh conversation
    function newConversation() {
        setCurrConversation(null);
        setMessages([]);
    }

    // get the currently selected conversation
    const currentConversation = conversations.find((conversation) => conversation.conversation_id === currConversation);

    const promptRef = useRef<HTMLTextAreaElement>(null);

    // change the textbox of the prompt as screen size changes
    function handlePromptChange(event: React.ChangeEvent<HTMLTextAreaElement>)
    {
        const textBox = event.currentTarget;
        textBox.style.height = "auto";
        textBox.style.height = `${textBox.scrollHeight}px`;
    }

    // send the prompt to the backend chat agent
    async function sendPrompt() {
        const textBox = promptRef.current;
        const text = textBox?.value.trim();
        if (!textBox || !text || sending) return;

        setMessages(prev => [...prev, { role: "user", content: text }]);
        textBox.value = "";
        textBox.style.height = "auto";
        setSending(true);

        let reply = "";
        try {
            const token = localStorage.getItem("_token");
            const headers = {
                "Content-Type": "application/json",
                Authorization: `Bearer ${token}`,
            };

            // create a conversation first if none is selected
            let conversationId = currConversation;
            if (!conversationId) {
                const newConv = await fetch(`${API_URL}/conversations`, { method: "POST", headers });
                if (!newConv.ok) throw new Error(`Could not create conversation (status ${newConv.status})`);
                conversationId = (await newConv.json()).conversation_id as string;
                setCurrConversation(conversationId);
            }

            const response = await fetch(`${API_URL}/conversations/${conversationId}/message`, {
                method: "POST",
                headers,
                body: JSON.stringify({ message: text }),
            });
            if (!response.ok) throw new Error(`Message failed (status ${response.status})`);
            const data = await response.json();
            reply = data.reply;
        } catch (err) {
            reply = `Error: ${err instanceof Error ? err.message : "something went wrong"}`;
        }

        setMessages(prev => [...prev, { role: "assistant", content: reply }]);
        setSending(false);
        void loadConversations(); // refresh sidebar (new chat / updated title)
    }

    // Enter sends, Shift+Enter adds a new line
    function handleKeyDown(event: React.KeyboardEvent<HTMLTextAreaElement>)
    {
        if (event.key === "Enter" && !event.shiftKey) {
            event.preventDefault();
            void sendPrompt();
        }
    }

    return (
        <div className="relative flex min-h-screen items-start justify-start bg-black">
            {/* Left panel */} 
            <section id="leftPanel" className="flex min-h-screen w-90 flex-col items-start pt-24 border-r-2 border-darkgray">
                <div className="absolute left-6 top-6">
                    <p className="text-4xl font-bold text-gold">ModelAdvisor</p>
                    <div className="mt-1 h-1 w-30 bg-ltgray" />
                </div>

                {/* HTML for new project button */} 
                <div>
                    <button onClick={newConversation} className="mx-auto text-4x1 font-semibold text-white hover:opacity-90 m-3 p-3">
                        <span className="inline-flex w-8 h-8 p-2 items-center justify-center rounded-full bg-medgray text-black mr-2">&#43;</span> 
                        New Project
                    </button>
                </div>

                {/* HTML for listing previous conversations */} 
                <div className="flex flex-col items-start ml-5">
                    <h2 className="text-2xl font-bold text-medgray">Recent Chats</h2>
                    {conversations.map((conversation) => (
                        <button
                            key={conversation.conversation_id}
                            onClick={() => openConversation(conversation.conversation_id)}
                            className={`text-left mt-2 hover:opacity-80 ${
                                conversation.conversation_id === currConversation ? "text-gold" : "text-white"
                            }`}
                        >
                            {conversation.title || "Untitled chat"}
                        </button>
                    ))}
                </div>

                {/* HTML for user area on dialog page */} 
                <div className="mt-auto flex w-full items-center justify-start border-t-2 border-darkgray p-5">
                    <div className="w-12 h-12 bg-gold rounded-full mr-5 p-6"></div>
                    <p className="text-white">{error || user?.email || "Loading user..."}</p>
                </div>
            </section>
            
            {/* Right panel */} 
            <section id="rightPanel" className="flex min-h-screen flex-1 flex-col w-auto items-center pt-10">
                <h1 className="text-4xl font-bold text-white">{currentConversation?.title ?? "New Conversation"}</h1>

                {/* Chat messages (TEMP) */}
                <div className="flex flex-col gap-3 w-full max-w-4xl my-6 px-4 overflow-y-auto">
                    {messages.map((message, i) => (
                        <div
                            key={i}
                            className={`p-3 rounded-lg max-w-[75%] whitespace-pre-wrap ${
                                message.role === "user"
                                    ? "self-end bg-gold text-black"
                                    : "self-start bg-darkgray text-white"
                            }`}
                        >
                            {message.content}
                        </div>
                    ))}
                    {sending && <div className="self-start text-medgray">Thinking…</div>}
                </div>

                <textarea ref={promptRef} onChange={handlePromptChange} onKeyDown={handleKeyDown} rows={1} placeholder="Enter prompt here" className="w-[90%] max-w-4xl h-20 bg-ltgray rounded-lg p-6 text-black"></textarea>
                <p className="text-white">ModelAdvisor is AI and can make mistakes. Please double check responses.</p>
            </section>
        </div>
    )
}