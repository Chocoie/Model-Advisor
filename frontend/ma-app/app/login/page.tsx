"use client";
import { useState, type ChangeEvent, type FormEvent } from 'react';
import { useRouter } from 'next/navigation';
import Link from "next/link";

export default function SignUp() 
{
    // need to connect to backend - curently are placeholders
    const[email, setEmail] = useState("");
    const[password, setPassword] = useState("");
    //const[userType, setUserType] = useState("");
    const[error, setError] = useState("");

    const router = useRouter();

    // log in user by sending user info to backend
    async function loginUser(): Promise<Response> 
    {
        // fetch
        return fetch("http://localhost:8000/auth/login", { //will need to change url later based on where web app is being hosted
            method: "POST",
            headers: {
                'Accept': 'application/json',
                'Content-Type': 'application/json'
            },
            body: JSON.stringify ({ email, password })
        });        
    }

    // check that email entered was actually an email
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    function isValidEmail(email: string): boolean 
    {
        return emailRegex.test(email);
    }

    // redirect user to main dialog page
    async function redirectUser(event: FormEvent<HTMLFormElement>) 
    {
        event.preventDefault();
        setError("");
                
        // ensure user has input all information needed
        if(!email || !password )
        {
            setError("All fields are required!")
            return;
        }

        // check if email is in valid format
        if(!isValidEmail(email))
        {
            setError("Must enter an email address. Try again.");
            return;
        }

        try {
            const response = await loginUser();

            // check if email was taken
            if(response.status === 401)
            {
                setError("Invalid email or password. Try again")
                return;
            }

            // check if response was successful
            if(!response.ok) {
                setError("Account login error. Please try again");
                return;
            }

            // stores uType and user id in local storage
            const userData = await response.json();
            localStorage.setItem('_email', userData.email);
            localStorage.setItem('_token', userData.token);

            router.push("/dialog")
        } catch {
            setError("Could not connect to the server. Please try again");
        }
        
    }

    // functions to set the actual user data
    function handleEmailChange(event: ChangeEvent<HTMLInputElement>) 
    {
        setEmail(event.target.value);
        setError("");
    }

    function handlePasswordChange(event: ChangeEvent<HTMLInputElement>)
    {
        setPassword(event.target.value);
    }

    return (
        <div className="relative flex min-h-screen items-center justify-center bg-black p-6">
            <div className="absolute left-6 top-6">
                <p className="text-4xl font-bold text-gold">ModelAdvisor</p>
                <div className="mt-1 h-1 w-30 bg-ltgray" />
            </div>
            
            <main className='mx-auto mt-10 flex w-full max-w-md flex-col items-center rounded-[2.5rem] bg-darkgray px-8 py-12'>
                <h2 className='mb-8 text-2xl font-bold text-white'>Welcome Back!</h2>
                {error && (
                    <p className='text-sm text-red-500 font-bold'>Error: {error}</p>
                )}
                <form onSubmit={redirectUser} className="flex w-full max-w-xs flex-col">
                    <div className='flex flex-col gap-0.5'>
                        <label htmlFor="email" className="text-sm font-bold text-white mg-0">Email</label>
                        <input onChange={handleEmailChange} type="email" className="h-10 w-full bg-ltgray px-2 text-black outline-none hover:opacity-90" required/>
                    </div>
                    <br/>
                    <div className='flex flex-col gap-0.5'>
                        <label>Password</label>
                        <input onChange={handlePasswordChange} type="password" maxLength={72} className="h-10 w-full bg-ltgray px-2 text-black outline-none hover:opacity-90" required/>
                    </div>
                    <br/>
                    <p className='text-center'>Don't have an account? <Link href="/signup" className="text-xl font-semibold text-gold hover:opacity-90">Sign Up</Link></p>
                    <button type='submit' className="mx-auto mt-5 w-40 h-12 rounded-full bg-gold px-6 py-3 text-lg font-semibold text-black hover:opacity-90">Login</button>
                </form>
            </main>
        </div>
    )
}