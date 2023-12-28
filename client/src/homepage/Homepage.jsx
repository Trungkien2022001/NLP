import React from 'react'
import { Predict } from './predict/Predict'
import './Homepage.scss'
import { Word2vec } from './word2vec/Word2vec';
export const Homepage = () => {

    return (

        <div className='main'>
            <div className="container">
                <Predict></Predict>
                <Word2vec></Word2vec>
            </div>
        </div>
    )
}
