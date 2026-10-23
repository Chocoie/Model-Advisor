"use client";
import { useEffect, useState, useRef } from "react";

type CurrentUser = {
    email: string;
};

type Conversation = {
    conversation_id: string;
    title: string;
}

export default function Dialog() {
    const [user, setUser] = useState<CurrentUser | null>(null);
    const [error, setError] = useState("");
    const [conversations, setConversations] = useState<Conversation[]>([]);
    const [currConversation, setCurrConversation] = useState<string | null>(null);

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
            } catch {
                setError("Could not connect to the server. Please try again.");
            }
        }

        void loadUser();
    }, []);

    // get id of current converstation
    const currentConversation = conversations.find((conversation) => conversation.conversation_id);

    const promptRef = useRef<HTMLTextAreaElement>(null);

    // change the textbox of the prompt as screen size changes
    function handlePromptChange(event: React.ChangeEvent<HTMLTextAreaElement>)
    {
        const textBox = event.currentTarget;
        textBox.style.height = "auto";
        textBox.style.height = `${textBox.scrollHeight}px`;
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
                    <button className="mx-auto text-4x1 font-semibold text-white hover:opacity-90 m-3 p-3">
                        <span className="inline-flex w-8 h-8 p-2 items-center justify-center rounded-full bg-medgray text-black mr-2">&#43;</span> 
                        New Project
                    </button>
                </div>

                {/* HTML for listing previous conversations */} 
                <div className="flex flex-col items-start ml-5">
                    <h2 className="text-2xl font-bold text-medgray">Recent Chats</h2>
                    {conversations.map((conversation) => (
                        <button key={conversation.conversation_id} onClick={() => setCurrConversation(conversation.conversation_id)} className="text-left text-white">
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
                <p></p>
                <p></p>
                <textarea ref={promptRef} onChange={handlePromptChange} rows={1} placeholder="Enter prompt here" className="w-300 h-20 bg-ltgray rounded-lg mr-5 p-6 text-black"></textarea>
                <p>ModelAdvisor is AI and can make mistakes. Please double check responses.</p>
            </section>
        </div>
    )
} 