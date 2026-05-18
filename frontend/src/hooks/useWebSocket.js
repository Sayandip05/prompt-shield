import { useEffect, useRef, useCallback } from 'react'

/**
 * Minimal WebSocket hook — connects when a URL is provided, disconnects on cleanup.
 * @param {string|null} url  WebSocket URL, or null to skip connecting
 * @param {function}    onMessage  Callback for incoming messages
 */
export function useWebSocket(url, onMessage) {
  const wsRef = useRef(null)

  const connect = useCallback(() => {
    if (!url) return
    const ws = new WebSocket(url)
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)
        onMessage?.(data)
      } catch {
        onMessage?.(event.data)
      }
    }
    ws.onerror = () => {} // suppress unhandled errors
    wsRef.current = ws
  }, [url, onMessage])

  useEffect(() => {
    connect()
    return () => wsRef.current?.close()
  }, [connect])

  const send = useCallback((data) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(data))
    }
  }, [])

  return { send }
}

export default useWebSocket
