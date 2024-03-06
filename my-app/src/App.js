import React from 'react';
import './App.css';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import ChatApp from './components/ChatApp';
import RegistPolicy from './components/RegistPolicy';
import OnBoardingStartPage from './components/OnBoarding';
import SelectPage from './components/SelectPage';
import LoadingPage from './components/LoadingPage';

function App() {
  return (
    <div className="App">
      <Routes>
        <Route path='/' element={<OnBoardingStartPage />} />
        <Route path='/chat' element={<ChatApp />} />
        <Route path='/regist' element={<RegistPolicy />} />
        <Route path='/select' element={<SelectPage />} />
        <Route path='/loading' element={<LoadingPage />} />

      </Routes>
    </div>
  );
}

export default App;
