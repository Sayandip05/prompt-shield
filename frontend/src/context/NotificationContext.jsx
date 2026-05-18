import { createContext, useContext } from 'react'

const NotificationContext = createContext(null)

// Minimal provider — expands when WebSocket notifications are wired up
export function NotificationProvider({ children }) {
  return (
    <NotificationContext.Provider value={{ notifications: [] }}>
      {children}
    </NotificationContext.Provider>
  )
}

export function useNotifications() {
  return useContext(NotificationContext)
}

export default NotificationContext
