import React from 'react';
import { Paper, Typography } from '@mui/material';

const Message = ({ text, role }) => {
  const isUser = role === 'user';
  const isChatbot = role === 'chatbot';

  return (
    <div style={{ display: 'flex', justifyContent: isUser ? 'flex-end' : 'flex-start', marginBottom: '10px' }}>
      <div style={{display: 'flex', flexDirection: 'column', alignItems: isUser ? 'flex-end' : 'flex-start', width: isUser ? '80%' : '100%' }}>
        <Paper elevation={0} style={{ marginBottom: '5px', backgroundColor: isUser ? '#DCF8C6' : '#E3F2FD' }}>
          <Typography variant="body1" style={{marginLeft:'20px', marginRight:'20px' ,marginTop:'8px', marginBottom:'8px', textAlign: 'left', padding: '10px' }}>{text}</Typography>
        </Paper>
      </div>
    </div>
  );
};

export default Message;
