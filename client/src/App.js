import {BrowserRouter, Routes, Route} from 'react-router-dom'
import './app.scss'
import { Homepage } from './homepage/Homepage';
function App() {
  return (
    <div className="App">
      <BrowserRouter>
        <Routes>
          <Route path='/' element = {<Homepage/>}/>
        </Routes>
      </BrowserRouter>
    </div>
  );
}

export default App;
