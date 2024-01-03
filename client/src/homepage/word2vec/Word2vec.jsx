import React, { useState } from "react";
import axios from "axios";
import "./Word2vec.scss";
import { Button, TextField } from "@mui/material";
import CSVReader from "react-csv-reader";
import Papa from "papaparse";
import { toast } from "react-toastify";
import { Loading } from "../loading/Loading";
export const Word2vec = () => {
  const host = process.env.REACT_APP_HOST || `http://localhost:5000`;

  const [ip, setIp] = useState("");
  const [result, setResult] = useState("");
  const [jsonData, setJsonData] = useState(null);
  const [loading, setLoading] = useState(false);
  const handleSubmit = async () => {
    await get();
  };
  console.log(process.env);
  async function get() {
    console.log("ping");
    const test = await axios.post(`${host}/model`, {
      input: ip,
    });
    console.log(test.data);
    if (!test.data.result) {
      toast.error(
        "Từ bạn tìm kiếm không có trong từ điển, vui lòng tìm với từ khác nhé"
      );
    }
    setResult(test.data.result);
    return test.data.result;
  }
  return (
    <div>
      {loading && <Loading></Loading>}
      <div className="title-model">2. Model Word2Vec</div>
      <div>
        <div className="chat-input">
          <TextField
            style={{ width: "100%" }}
            className="text-input text-input-95"
            id="standard-basic"
            // multiline
            //   maxRows={3}
            label="Từ"
            variant="outlined"
            placeholder="Nhập từ ngữ cần tìm kiếm, ví dụ: Tốt, xấu"
            onChange={(e) => setIp(e.target.value)}
          />
        </div>
        <div className="container-btn">
          <div className="submit">
            <Button onClick={() => handleSubmit()} variant="contained">
              Submit
            </Button>
          </div>
        </div>
        <div className="result-header">Kết quả về độ tương đồng</div>

        <hr />

        <div className="result-container">
          <div className="result">
            {result && result.length ? (
              result.map((item, index) => (
                <div className="item" key={index}>
                  <div className="text">{item[0]}</div>
                  <div className="label">{(item[1] * 100).toFixed(2)}%</div>
                </div>
              ))
            ) : (
              <></>
            )}
          </div>
        </div>
        <hr />
      </div>
    </div>
  );
};
