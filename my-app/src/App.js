import React from 'react';
import './App.css';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import ChatApp from './components/ChatApp';
import OnBoardingStartPage from './components/OnBoarding';
import SelectPage from './components/SelectPage';
import LoadingPage from './components/LoadingPage';
import RecommendPage from './components/RecommendPage';
import ChatAppV2 from './components/ChatAppV2';


function App() {
  return (
    <div className="App">
      <Routes>
        <Route path='/' element={<OnBoardingStartPage />} />
        <Route path='/chat' element={<ChatApp />} />
        <Route path='/select' element={<SelectPage />} />
        <Route path='/loading' element={<LoadingPage />} />
        <Route path='/recommendPage' element={<RecommendPage />} />
        <Route path="/chatv2" element={<ChatAppV2 />} />
      </Routes>
    </div>
  );
}

export default App;
