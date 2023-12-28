import { CircularProgress } from '@mui/material'
import React from 'react'
import './Loading.scss'

export const Loading = ({ style }) => {
    const customStyle = {
        size: 40,
        ...style
    }
    return (
        <div className='test'>
            {/* <div style={{textAlign: 'center', width: "100%", fontSize: "25px"}}>
                Đang xử lý
            </div> */}
            <div className='loading-container'>
                <CircularProgress
                    size={customStyle.size}
                    style={{ margin: "auto" }}
                    
                />
            </div>
        </div>
    )
}
