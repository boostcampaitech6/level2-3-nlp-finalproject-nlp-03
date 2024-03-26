import React from 'react';
import './App.css';
import ChatApp from './components/ChatApp';

function App() {
  return (
    <div className="App">
      {/* <header className="App-header">
        <h1>React 채팅 앱</h1>
      </header> */}
      <main className="App-main">
        <ChatApp />
      </main>
    </div>
  );
}

export default App;
