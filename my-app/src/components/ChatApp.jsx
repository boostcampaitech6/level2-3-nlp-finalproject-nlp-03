import React, { useState, useEffect, useRef } from 'react';
import { Container, Typography, TextField, Button, Grid } from '@mui/material';
import Message from './Message';
import SourceDocument from './SourceDocument';

const ChatApp = () => {
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [isResponse, setIsResponse] = useState(true);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleMessageSend = async () => {
    if (inputValue.trim() !== '') {
      // 사용자가 보낸 메시지를 추가
      setMessages([...messages, { text: inputValue, role: 'user' }]);
      setInputValue('');
      // 서버 요청 보내고 응답 받기 (시뮬레이션)
      try {
        const response = await fetch('http://localhost:8000/query', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            "query": inputValue // inputValue를 그대로 보냅니다.
          }) // inputValue를 그대로 보냅니다.
        });
        const data = await response.json();

        // 서버로부터 받은 응답을 채팅창에 추가
        setMessages(messages => [
          ...messages, 
          { text: data.response, role: 'chatbot' },
          { text: data.source_document, role: 'source'}
        ]);
        console.log(messages)
      } catch (error) {
        console.error('Error sending message:', error);
      }
      setIsResponse(true);
      setInputValue('');
    }
  };

  const handleInputChange = (e) => {
    if (isResponse == false) {
      return
    }
    setInputValue(e.target.value);
  };

  const handleKeyDown = (e) => {
    if (e.nativeEvent.isComposing) return
    if (e.key === 'Enter') {
      setIsResponse(false);
      handleMessageSend();
    }
  };

  return (
    <Container maxWidth="md" style={{ marginTop: '50px' }}>
      <Typography variant="h4" component="h1" align="center" gutterBottom>
        👨‍👩‍👧‍👦 동행 챗봇 prototype
      </Typography>
      <Grid container spacing={2}>
        <Grid item xs={12}>
          <div style={{ minHeight: '80vh', maxHeight: '80vh', overflowY: 'auto' }}>
            {messages.map((message, index) => (
                message.role === 'source' ? (
                  <SourceDocument key={index} text={message.text} />
                ) : (
                  <Message key={index} text={message.text} role={message.role} />
                )
              ))}
            <div style={{ float: 'right', width: '8px', height: '100%', marginRight: '-8px' }} />
            <div ref={messagesEndRef} />
          </div>
        </Grid>
        <Grid item xs={12}>
          <Grid container spacing={2} alignItems="flex-end" justifyContent="flex-end">
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="질문을 입력해주세요."
                variant="outlined"
                value={inputValue}
                onKeyDown={handleKeyDown}
                onChange={handleInputChange}
              />
            </Grid> 
          </Grid>
        </Grid>
      </Grid>
    </Container>
  );
};

export default ChatApp;
