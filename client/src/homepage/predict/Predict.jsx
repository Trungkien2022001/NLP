import React, { useState } from 'react'
import axios from 'axios'
import './Predict.scss'
import { Button, TextField } from '@mui/material'
import CSVReader from 'react-csv-reader';
import { toast } from 'react-toastify';
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
    const [length, setLength] = useState(0);
    const [l, setL] = useState(0)
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
            setL(test.data.result.length)
            setIp([])
            setLength(0)
        }
        setLoading(false)
    }
    const handleCsvData = async result => {
        console.log(result.data.map(i => i.content))
        const data = result.data.map(i => i.content).filter(i=> !!i)
        setIp(data)
        setLength(data.length)
        if(!data.length){
            toast.error("Có vẻ như bạn không nhập đúng định dạng file CSV. Tên cột trong sheet là content chứa danh sách các câu văn nhé",  {
                position: toast.POSITION.TOP_CENTER
              })
        }

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
                        placeholder='Nhập câu văn cần phân loại, ví dụ: Sản phẩm này rất tốt'
                        onChange={e => setIp([e.target.value])}
                    />
                    {/* <input style={{padding: "10px"}} type="file" id="fileInput" className="file-input" onChange={handleFileSelect} /> */}
                    <div style={{ margin: "10px 0px" }}>

                        <label htmlFor="fileInput" className="file-label">
                            <span>Nhập từ tệp csv </span>
                            <input
                                style={{ padding: '10px', display: 'none' }}
                                type="file"
                                id="fileInput"
                                className="file-input"
                                onChange={handleFileSelect}
                                placeholder='VUi lòng nhập file có định dạng csv với tên cốt là content'
                            />
                        </label>
                        {
                            length ? <span>{`Có ${length} câu văn được tải lên`}</span> : <></>
                        }
                    </div>
                </div>
                <div className="submit">
                    <Button onClick={() => handleSubmit()} variant="contained">Submit</Button>
                </div>
                <div className="result-header">
                    Kết quả
                </div>
                {
                    l ? <div>{`Đã phân tích thành công ${l} câu văn!`}</div> : <></>
                }
                <hr />
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
            <hr />
        </div>

    )
}
