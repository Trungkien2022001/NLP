import React, { useState } from 'react'
import axios from 'axios'
import './Predict.scss'
import { Button, TextField } from '@mui/material'
import CSVReader from 'react-csv-reader';
import Papa from 'papaparse';
import { Loading } from '../loading/Loading';
const labelKey = {
    "Tích cực": "pos",
    "Tiêu cực": "neg",
    "Trung tính": "neu",
}

export const Predict = () => {

    const [ip, setIp] = useState([])
    const [result, setResult] = useState("")
    const [jsonData, setJsonData] = useState(null);
    const [loading, setLoading] = useState(false)
    const handleSubmit = async () => {
        setResult([])
        console.log("Getting data!!", ip.length)
        setLoading(true)
        if (ip.length) {
            const test = await axios.post(`http://localhost:5000`, {
                input: ip
            })
            console.log(test.data)
            setResult(test.data.result)
            setIp([])
        }
        setLoading(false)
    }
    async function get() {
        setLoading(true)
        if (!ip.length) {
            const test = await axios.post(`http://localhost:5000`, {
                input: ip
            })
            console.log(test.data)
            setResult(test.data.result)
            setIp([])
        }
        setLoading(false)


    }
    const handleCsvData = result => {
        console.log(result.data.map(i => i.content))
        setIp(result.data.map(i => i.content))
        // await handleSubmit()
    }
    const handleFileSelect = event => {
        const file = event.target.files[0]
        Papa.parse(file, {
            complete: handleCsvData,
            header: true
        })
    }
    return (
        <div>
            <div className="header">
                Bài toán phân loại cảm xúc
            </div>
            <div className="title">
                1. Phân loại
            </div>
            <div style={{ position: "relative" }}>

                {loading && <Loading></Loading>}
                <div className="chat-input">
                    <TextField
                        style={{ width: "100%" }}
                        className='text-input text-input-95'
                        id="standard-basic"
                        // multiline
                        //   maxRows={3}
                        label="Câu văn"
                        variant="outlined"
                        placeholder='Nhập câu văn cần phân loại'
                        onChange={e => setIp([e.target.value])}
                    />
                    {/* <input style={{padding: "10px"}} type="file" id="fileInput" className="file-input" onChange={handleFileSelect} /> */}
                    <div style={{ margin: "20px 0px" }}>

                        <label htmlFor="fileInput" className="file-label">
                            <span>Nhập nhiều câu văn từ tệp</span>
                            <input
                                style={{ padding: '10px', display: 'none' }}
                                type="file"
                                id="fileInput"
                                className="file-input"
                                onChange={handleFileSelect}
                            />
                        </label>
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
                                        {item.key}
                                    </div>
                                    <div className={labelKey[item.label]}>
                                        {item.label}
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
