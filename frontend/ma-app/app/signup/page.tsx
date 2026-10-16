"use client";
import { useState } from 'react';
import { useRouter } from 'next/navigation';
import Link from "next/link";
import axios from 'axios';

export default function SignUp() 
{
    // need to connect to backend - curently are placeholders
    const[email, setEmail] = useState("");
    const[password, setPassword] = useState("");
    const[password2, setPassword2] = useState("");
    const[userType, setUserType] = useState("");
    const[isEmailTaken, setIsEmailTaken] = useState(false);
    const[error, setError] = useState("");

    // add new user by sending user info to backend
    async function addUser()
    {
        // may need to change later based on backend json
        let doc = {
            email: email,
            password: password,
            uType: userType
        }

        // fetch
        let response = await fetch("http://localhost:8000", { //will need to change url later based on where web app is being hosted
            method: "POST",
            body: JSON.stringify(doc),
            headers: {
                'Accept': 'application/json',
                'Content-Type': 'application/json'
            }
        });

        // check if response was successful
        if(response.ok) {
            const userData = await response.json();
            return {
                _id: userData._id,
                userType: doc.uType
            }
        } else {
            return null;
        }
    }

    // check that email entered was actually an email
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    function isValidEmail(email: string): boolean 
    {
        return emailRegex.test(email);
    }

    // check if email was taken
    const checkEmail = async() => {
        try {
            // will probably have to change based on db and backend
            const response = await axios.get('http://localhost:8000/users/checkEmail', {params: {email: email}});
            const taken = response.data.isTaken;
            setIsEmailTaken(taken);
            return taken;
        } catch (error) {
            return true;
        }
    };

    // check if password meets password requirements
            // must have captial letter, a number, special character, and at least 8 characters long
    const passwordRegex = /^(?=.*[A-Z])(?=.*\d)(?=.*[^A-Za-z0-9]).{8,}$/;
    function testPassword(password: string): boolean 
    {
        return passwordRegex.test(password);
    }

    // redirect user to main dialog page
    async function redirectUser(event) 
    {
        event.preventDefault();
        setError("");
        
        // ensure user has input all information needed
        if(!email || !password || !password2)
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

        // check if email is already taken
        const isTaken = await checkEmail();
        if(isTaken)
        {
            setError("Email is already taken. Try Again.");
            return;
        }

        // check if passwords match
        if(password !== password2)
        {
            setError("Passwords do not match. Please try again.");
            return;
        }

        // validates password
        if(!testPassword(password))
        {
            setError("Password must contain a captial letter, a number, a special character, and be 8 characters long. Try again.");
            return;
        }

        // stores uType and user id in local storage
        const userData = await addUser();
        const router = useRouter();
        if(typeof userData === 'object' && userData !== null)
        {
            localStorage.setItem('userType', userData.userType);
            localStorage.setItem('_id', userData._id);

            router.push("/dialog")
        } else {
            setError("Account creation error!");
        }
    }

    // functions to set the actual user data
    function handleEmailChange(data) 
    {
        setEmail(data.target.value);
        setIsEmailTaken(false);
        setError("");
    }

    function handlePasswordChange(data)
    {
        setPassword(data.target.value);
    }

    function handlePassword2Change(data)
    {
        setPassword2(data.targe.value);
    }


    return (
        <div className="relative flex min-h-screen items-center justify-center bg-black p-6">
            <div className="absolute left-6 top-6">
                <p className="text-4xl font-bold text-gold">ModelAdvisor</p>
                <div className="mt-1 h-1 w-30 bg-ltgray" />
            </div>
            
            <main className='mx-auto mt-10 flex w-full max-w-md flex-col items-center rounded-[2.5rem] bg-darkgray px-8 py-12'>
                <h2 className='mb-8 text-2xl font-bold text-white'>Create an account!</h2>
                {error && (
                    <p className='text-sm text-red-500 font-bold'>Error: {error}</p>
                )}
                <form className="flex w-full max-w-xs flex-col">
                    <div className='flex flex-col gap-0.5'>
                        <label htmlFor="email" className="text-sm font-bold text-white mg-0">Email</label>
                        <input onChange={handleEmailChange} onBlur={() => checkEmail(email)} type="email" className="h-10 w-full bg-ltgray px-2 text-black outline-none hover:opacity-90" required/>
                    </div>
                    <br/>
                    <div className='flex flex-col gap-0.5'>
                        <label>Password</label>
                        <input onChange={handlePasswordChange} type="password" className="h-10 w-full bg-ltgray px-2 text-black outline-none hover:opacity-90" required/>
                    </div>
                    <br/>
                    <div className='flex flex-col gap-0.5'>
                        <label>Confirm Password</label>
                        <input onChange={handlePassword2Change} type="password" className="h-10 w-full bg-ltgray px-2 text-black outline-none hover:opacity-90" required/>
                    </div>
                    <br/>
                    <p className='text-center'>Already have an account? <Link href="/login" className="text-xl font-semibold text-gold hover:opacity-90">Login</Link></p>
                    <button onClick={redirectUser} className="mx-auto mt-5 w-40 h-12 rounded-full bg-gold px-6 py-3 text-lg font-semibold text-black hover:opacity-90">Sign Up</button>
                </form>
            </main>
        </div>
    )
}