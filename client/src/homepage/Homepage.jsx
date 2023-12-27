import React, { useEffect, useState } from 'react'
import axios from 'axios'
export const Homepage = () => {
    const [result, setResult] = useState("")
    const [ip, setIp] = useState("")
    // useEffect(() => {
    //     get()
    // }, [])
    async function get() {
        const test = await axios.post(`http://localhost:5000`, {
            input: ip
        })
        console.log("ping")
        console.log(test.data)
        setResult(test.data.result)
        return test.data.result
    }
    const handleSubmit= async ()=>{
        await get() 
    }
    return (

        <div className='main'>
            <div className="chatinput">
                <input
                    type="text"
                    // onKeyDown={onEnterPress} 
                    placeholder='Nhập tin nhắn'
                    value={ip}
                    onChange={e => setIp(e.target.value)} />
            </div>
            <button onClick={handleSubmit}>get</button>
            <div className="result">
                {result}
            </div>
        </div>
    )
}
