import { Link } from 'react-router-dom'
import { ShieldOff, LogIn } from 'lucide-react'

const UnauthorizedPage = () => {
  return (
    <div className="min-h-screen bg-gray-50 flex items-center justify-center px-4">
      <div className="text-center max-w-md">
        <div className="w-20 h-20 bg-red-100 rounded-2xl flex items-center justify-center mx-auto mb-6">
          <ShieldOff className="w-10 h-10 text-red-500" />
        </div>
        <h1 className="text-3xl font-bold text-gray-900 mb-3">Access Denied</h1>
        <p className="text-gray-600 mb-8">
          You don't have permission to view this page. Please sign in with the correct account or go back to safety.
        </p>
        <div className="flex flex-col sm:flex-row gap-3 justify-center">
          <Link to="/login" className="btn-primary flex items-center gap-2">
            <LogIn className="w-4 h-4" />
            Sign In
          </Link>
          <Link to="/" className="btn-secondary">
            Go to Home
          </Link>
        </div>
      </div>
    </div>
  )
}

export default UnauthorizedPage
