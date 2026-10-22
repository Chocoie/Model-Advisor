import Link from "next/link";

export default function Landing() 
{
    return (
        <div className="flex h-screen flex-col">
            <nav className="flex justify-end px-12 py-10">
                <Link href="/login" className="text-xl font-semibold text-gold">Login</Link>
            </nav>
            
            <main className="flex flex-1 flex-col justify-center px-20 pb-48">
                <div className="flex max-h-screen flex-col justify-center">
                    <h1 className="text-8xl font-bold tracking-tight text-gold">ModelAdvisor</h1>
                    <p className="text-3xl tracking-tight text-white max-w-160">A conversational platform for goal-specific evaluation and recommendation of large language models</p>
                    <Link href="/signup" className="mt-5 w-fit rounded-full bg-gold px-6 py-3 text-lg font-semibold text-black hover:opacity-90">Get Started &rarr;</Link>
                </div>
                <div>

                </div>
            </main>
        </div>
    )
}