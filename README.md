# ModelAdvisor
To currently run ModelAdvisor locally do the following steps:
1. follow `backend/README.md` to install the appropriate requirements
2. run `pip install npm`
3. `cd backend` and run the `uvicorn main:app --reload` command
4. in another terminal `cd frontend/ma-app` and run `npm run dev`

Ensure that you create a `.env.local` file and put the follow into it
`NEXT_PUBLIC_API_URL=http://localhost:8000`

You should be able to test the POST response stated in the backend README.md