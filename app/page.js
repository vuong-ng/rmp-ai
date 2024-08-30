'use client'
import Image from "next/image";
import { useState } from 'react'
import { Button, TextField, Stack, Box, useColorScheme} from '@mui/material'
import Select from '@mui/material/Select';
import MenuItem from '@mui/material/MenuItem';
import { ThemeProvider, createTheme } from '@mui/material/styles'

function MyApp() {
  const { mode, setMode } = useColorScheme()
  if (!mode) {
    return null;
  }
  return (
    <Box
      sx={{
        display: 'flex',
        width: '100%',
        alignItems: 'center',
        justifyContent: 'center',
        bgcolor: 'background.default',
        color: 'text.primary',
        borderRadius: 1,
        p: 3, minHeight: "56px",
      }}
    >
      <Select
        value={mode}
        onChange={(e) =>
          setMode(e.target.value)
        }>
        <MenuItem value="system">System</MenuItem>
        <MenuItem value="light">Light</MenuItem>
        <MenuItem value="dark">Dark</MenuItem>
      </Select>
    </Box>
  );
}
const lightTheme = createTheme({
  colorSchemes: {
    light:true
  }
})

export default function Home() {
  const { mode, setMode } = useColorScheme()
  
  const [messages, setMessages] = useState([
    {
      role: "assistant",
      content:"Hi! I'm the Rate My Professor support assistant. How can I help you"
    },
  ])
  const [message, setMessage] = useState('')

  const sendMessage = async () => {
    setMessage('')
    setMessages((messages) => [
      ...messages,
      { role: 'user', content: message },
      {role:'assistant', content:''}
    ])
    const response = fetch('api/chat', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json' 
      },
      body: JSON.stringify([...messages,
        { role: 'user', content: message }
      ])
    }).then(async (res) => {
      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let result = ''
      return reader.read().then(function processText({ done, value }) {
        if (done) {
          return result
        }
        const text = decoder.decode(value || new Uint8Array(), { stream: true })
        setMessages((messages) => {
          let lastMessage = messages[messages.length - 1]
          let otherMessages = messages.slice(0, messages.length - 1)
          return [
            ...otherMessages,
            {...lastMessage,content: lastMessage.content +text}
          ]
        })
        return reader.read().then(processText)
      })
    })

  }
  return (
    <ThemeProvider theme={lightTheme}>
    <Box
      width="100vw"
      height="100vh"
      display='flex'
      flexDirection='column'
      justifyContent='center'
        alignItems='center'
      bgcolor='white'>
  
      <Stack 
        direction='column'
        width='500px'
        height='700px'
        border='1px solid black'
        p={2}
        spacing={3}
      >
        <Stack
        direction='column'
        spacing={2}
        flexGrow={1}
        overflow={'auto'}
        maxHeight={'100%'}>
        {
          messages.map((message, index) => (
            <Box
              key={index}
              display='flex'
              justifyContent=
              {message.role === 'assistant' ? 'flex-start' : 'flex-end'}
            >
              <Box
                bgcolor={
                  message.role === 'assitant' ? 'primary.main' : 'secondary.main'}
                color='white'
                borderRadius={16}
                p={3}
              >
                {message.content}
              </Box>
            </Box>
            
          ))}
        </Stack>
        <Stack
          direction='row' 
          spacing={2}
        >
          <TextField
            label="Message"
            fullWidth
            value={message}
            onChange={(e) => {
            setMessage(e.target.value)
            }} />
          <Button variant='contained' onClick={sendMessage}>
            Send
          </Button>
        </Stack>
    </Stack>
    </Box>
    </ThemeProvider>

  )
}
