import React, { useState } from 'react'
import axios from 'axios'
import './Word2vec.scss'
import { Button, TextField } from '@mui/material'
import CSVReader from 'react-csv-reader';
import Papa from 'papaparse';
import { Loading } from '../loading/Loading';

export const Word2vec = () => {

    const [ip, setIp] = useState("")
    const [result, setResult] = useState("")
    const [jsonData, setJsonData] = useState(null);
    const [loading, setLoading] = useState(false)
    const handleSubmit = async () => {
        await get()
    }
    async function get() {
        const test = await axios.post(`http://localhost:5000/model`, {
            input: ip
        })
        console.log("ping")
        console.log(test.data)
        setResult(test.data.result[0])
        return test.data.result


    }
    return (
        <div>
            <div className="title" style={{ marginTop: "50px" }}>
                2. Model Word2Vec
            </div>
            <div>
                {loading && <Loading></Loading>}

                <div className="chat-input">
                    <TextField
                        style={{ width: "100%" }}
                        className='text-input text-input-95'
                        id="standard-basic"
                        // multiline
                        //   maxRows={3}
                        label="Từ"
                        variant="outlined"
                        placeholder='Nhập từ ngữ cần tìm kiếm'
                        onChange={e => setIp(e.target.value)}
                    />
                    {/* <input style={{padding: "10px"}} type="file" id="fileInput" className="file-input" onChange={handleFileSelect} /> */}
                    <div style={{ margin: "20px 0px" }}>
                    </div>
                </div>
                <div className="submit">
                    <Button onClick={() => handleSubmit()} variant="contained">Submit</Button>
                </div>
                <div className="result-header">
                    Kết quả
                </div>
                <div className="result-container">
                    <div className="result">
                        {result && result.length ?
                            result.map((item, index) =>
                                <div className="item" key={index}>
                                    <div className="text">
                                        {item[0]}
                                    </div>
                                    <div className="label">
                                        {item[1] * 100}
                                    </div>
                                </div>
                            )
                            :
                            <></>}
                    </div>
                </div>
            </div>
        </div>

    )
}
