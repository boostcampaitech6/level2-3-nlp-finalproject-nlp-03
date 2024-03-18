import React, { useState, useEffect, useRef } from 'react';
import { Container, Typography, TextField, Button, Grid } from '@mui/material';
import Message from './Message';
import SourceDocument from './SourceDocument';
import { useLocation } from 'react-router-dom';

const ChatAppV2 = () => {
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [isResponse, setIsResponse] = useState(true);
  const messagesEndRef = useRef(null);
  const location = useLocation();
  const policy = location.state?.policy || {};
  const [chatbotType, setChatbotType] = useState('');

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

  const handleButtonClick = (type) => {
    setChatbotType(type);
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
      console.log('chatbotType:', chatbotType);
      console.log('policy:', policy);
      setIsResponse(false);
      handleMessageSend();
    }
  };

  return (
    <Container maxWidth="md" style={{ marginTop: '50px' }}>
      <Typography variant="h4" component="h1" align="center" gutterBottom>
        👨‍👩‍👧‍👦 {policy.PolicyName} 안내 챗봇
      </Typography>
      <Grid container spacing={2}>
        {/* chatbotType이 빈 문자열이 아닌 경우 */}
        {chatbotType !== '' ? (
          <>
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
          </>
        ) : (
          <Grid container spacing={2} justifyContent="center">
            <div style={{ paddingTop: '60px', minHeight: '60vh', maxHeight: '60vh', display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center' }}>
              <div style={{ marginBottom: '50px' }}>
                <Button
                  style={{
                    width: '200px',
                    height: '50px',
                    fontSize: '1rem',
                    fontWeight: 'bold',
                    borderRadius: '8px',
                    backgroundColor: chatbotType === '신청절차' ? '#1976D2' : '#2196F3',
                    color: '#FFFFFF',
                    border: 'none',
                    cursor: 'pointer',
                    transition: 'background-color 0.3s',
                  }}
                  onClick={() => handleButtonClick('신청절차')}
                >
                  신청절차 알아보기
                </Button>
              </div>
              <div style={{ marginBottom: '50px' }}>
                <Button
                  style={{
                    width: '200px',
                    height: '50px',
                    fontSize: '1rem',
                    fontWeight: 'bold',
                    borderRadius: '8px',
                    backgroundColor: chatbotType === '신청자격' ? '#388E3C' : '#4CAF50',
                    color: '#FFFFFF',
                    border: 'none',
                    cursor: 'pointer',
                    transition: 'background-color 0.3s',
                  }}
                  onClick={() => handleButtonClick('신청자격')}
                >
                  신청자격 확인하기
                </Button>
              </div>
              <div>
                <Button
                  style={{
                    width: '200px',
                    height: '50px',
                    fontSize: '1rem',
                    fontWeight: 'bold',
                    borderRadius: '8px',
                    backgroundColor: chatbotType === '정보제공' ? '#FFA000' : '#FFC107',
                    color: '#FFFFFF',
                    border: 'none',
                    cursor: 'pointer',
                    transition: 'background-color 0.3s',
                  }}
                  onClick={() => handleButtonClick('정보제공')}
                >
                  정보 제공받기
                </Button>
              </div>
            </div>
          </Grid>
        )}
      </Grid>
    </Container>
  );
};

export default ChatAppV2;
