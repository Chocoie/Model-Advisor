"use client";
import { useEffect, useState, useRef } from "react";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

type CurrentUser = {
    email: string;
};

type Conversation = {
    conversation_id: string;
    title: string;
};

type ChatMessage = {
    role: "user" | "assistant";
    content: string;
};

type ConversationHistory = {
    conversation_id: string;
    messages: ChatMessage[];
};

type ConversationList = {
    conversations: Conversation[];
};

type NewConversation = {
    conversation_id: string;
};

type MessageResponse = {
    reply: string;
    skills: string[];
    results: { model: string; avg_rank: number; scores: Record<string, number> }[];
};

async function apiRequest<T>(path: string, token: string, init: RequestInit = {}): Promise<T> {
    const headers = new Headers(init.headers);
    headers.set("Accept", "application/json");
    headers.set("Authorization", `Bearer ${token}`);
    if (init.body) {
        headers.set("Content-Type", "application/json");
    }

    const response = await fetch(`${API_BASE_URL}${path}`, { ...init, headers });
    if (!response.ok) {
        const responseText = await response.text();
        let message = response.statusText || `Request failed (${response.status})`;
        if (responseText) {
            try {
                const body = JSON.parse(responseText) as { detail?: string };
                message = body.detail ?? responseText;
            } catch {
                message = responseText;
            }
        }
        throw new Error(message);
    }
    return response.json() as Promise<T>;
}

export default function Dialog() {
    const [user, setUser] = useState<CurrentUser | null>(null);
    const [error, setError] = useState("");
    const [conversations, setConversations] = useState<Conversation[]>([]);
    const [currConversation, setCurrConversation] = useState<string | null>(null);
    const [messages, setMessages] = useState<ChatMessage[]>([]);
    const [draft, setDraft] = useState("");
    const [isSending, setIsSending] = useState(false);
    const [isLoadingConversation, setIsLoadingConversation] = useState(false);

    useEffect(() => {
        let cancelled = false;

        async function loadUser() {
            const token = localStorage.getItem("_token");
            if (!token) {
                if (!cancelled) {
                    setError("No login session found. Please log in again.");
                }
                return;
            }

            try {
                const [userData, conversationData] = await Promise.all([
                    apiRequest<CurrentUser>("/auth/me", token),
                    apiRequest<ConversationList>("/conversations", token),
                ]);
                if (!cancelled) {
                    setUser(userData);
                    setConversations(conversationData.conversations);
                }
            } catch (requestError) {
                if (!cancelled) {
                    const message = requestError instanceof Error ? requestError.message : "Could not connect to the server.";
                    setError(message === "Invalid token" ? "Your login session has expired. Please log in again." : message);
                }
            }
        }

        void loadUser();
        return () => {
            cancelled = true;
        };
    }, []);

    const promptRef = useRef<HTMLTextAreaElement>(null);
    const currentConversation = conversations.find((conversation) => conversation.conversation_id === currConversation);

    function handlePromptChange(event: React.ChangeEvent<HTMLTextAreaElement>) {
        const textBox = event.currentTarget;
        const maxHeight = 160;
        textBox.style.height = "auto";
        textBox.style.height = `${Math.min(textBox.scrollHeight, maxHeight)}px`;
        textBox.style.overflowY = textBox.scrollHeight > maxHeight ? "auto" : "hidden";
        setDraft(textBox.value);
    }

    async function createConversation() {
        const token = localStorage.getItem("_token");
        if (!token) {
            setError("No login session found. Please log in again.");
            return;
        }

        setError("");
        try {
            const created = await apiRequest<NewConversation>("/conversations", token, { method: "POST" });
            const conversationData = await apiRequest<ConversationList>("/conversations", token);
            setConversations(conversationData.conversations);
            setCurrConversation(created.conversation_id);
            setMessages([]);
        } catch (requestError) {
            setError(requestError instanceof Error ? requestError.message : "Could not create a conversation.");
        }
    }

    async function selectConversation(conversationId: string) {
        const token = localStorage.getItem("_token");
        if (!token) {
            setError("No login session found. Please log in again.");
            return;
        }

        setError("");
        setIsLoadingConversation(true);
        try {
            const history = await apiRequest<ConversationHistory>(
                `/conversations/${encodeURIComponent(conversationId)}`,
                token,
            );
            setCurrConversation(history.conversation_id);
            setMessages(history.messages);
        } catch (requestError) {
            setError(requestError instanceof Error ? requestError.message : "Could not load this conversation.");
        } finally {
            setIsLoadingConversation(false);
        }
    }

    async function sendMessage(event: React.FormEvent<HTMLFormElement>) {
        event.preventDefault();
        const message = draft.trim();
        if (!message || isSending) {
            return;
        }

        const token = localStorage.getItem("_token");
        if (!token) {
            setError("No login session found. Please log in again.");
            return;
        }

        setError("");
        setIsSending(true);
        try {
            let conversationId = currConversation;
            if (!conversationId) {
                const created = await apiRequest<NewConversation>("/conversations", token, { method: "POST" });
                conversationId = created.conversation_id;
                setCurrConversation(conversationId);
            }

            const result = await apiRequest<MessageResponse>(
                `/conversations/${encodeURIComponent(conversationId)}/message`,
                token,
                { method: "POST", body: JSON.stringify({ message }) },
            );
            setMessages((existing) => [
                ...existing,
                { role: "user", content: message },
                { role: "assistant", content: result.reply },
            ]);
            setDraft("");
            if (promptRef.current) {
                promptRef.current.style.height = "4rem";
                promptRef.current.style.overflowY = "hidden";
            }

            const conversationData = await apiRequest<ConversationList>("/conversations", token);
            setConversations(conversationData.conversations);
        } catch (requestError) {
            setError(requestError instanceof Error ? requestError.message : "Could not send your message.");
        } finally {
            setIsSending(false);
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
                    <button onClick={() => void createConversation()} className="mx-auto text-4x1 font-semibold text-white hover:opacity-90 m-3 p-3">
                        <span className="inline-flex w-8 h-8 p-2 items-center justify-center rounded-full bg-medgray text-black mr-2">&#43;</span> 
                        New Project
                    </button>
                </div>

                {/* HTML for listing previous conversations */} 
                <div className="flex flex-col items-start ml-5">
                    <h2 className="text-2xl font-bold text-medgray">Recent Chats</h2>
                    {conversations.map((conversation) => (
                        <button key={conversation.conversation_id} onClick={() => void selectConversation(conversation.conversation_id)} className="text-left text-white">
                            {conversation.title}
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
                {error && <p role="alert" className="mt-4 text-red-500">{error}</p>}
                <div className="my-8 flex w-full max-w-3xl flex-1 flex-col gap-4 overflow-y-auto px-6">
                    {isLoadingConversation ? (
                        <p className="text-white">Loading conversation...</p>
                    ) : messages.map((message, index) => (
                        <div key={`${index}-${message.role}`} className={`max-w-[80%] whitespace-pre-wrap rounded-lg p-4 text-black ${message.role === "user" ? "self-end bg-gold" : "self-start bg-ltgray"}`}>
                            {message.content}
                        </div>
                    ))}
                </div>
                <form onSubmit={sendMessage} className="mb-4 w-full max-w-3xl px-6">
                    <div className="relative w-full">
                        <textarea
                            ref={promptRef}
                            value={draft}
                            onChange={handlePromptChange}
                            rows={1}
                            placeholder="Enter prompt here"
                            className="block min-h-16 max-h-40 w-full resize-none rounded-lg bg-ltgray py-4 pl-4 pr-24 text-black overflow-y-hidden"
                        />
                        <button type="submit" disabled={isSending || !draft.trim()} className="absolute right-2 top-1/2 -translate-y-1/2 rounded-full bg-gold px-4 py-2 font-semibold text-black disabled:opacity-50">
                            {isSending ? "Sending..." : "Send"}
                        </button>
                    </div>
                </form>
                <p className="mb-4 w-full max-w-3xl px-6 text-left">ModelAdvisor is AI and can make mistakes. Please double check responses.</p>
            </section>
        </div>
    )
} 