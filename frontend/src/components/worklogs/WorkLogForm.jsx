import { useState } from 'react'

export default function WorkLogForm({ onSubmit, loading }) {
  const [form, setForm] = useState({ date: '', hours: '', description: '' })

  const handleSubmit = (e) => {
    e.preventDefault()
    onSubmit?.(form)
    setForm({ date: '', hours: '', description: '' })
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div className="grid grid-cols-2 gap-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Date</label>
          <input type="date" value={form.date} onChange={e => setForm({...form, date: e.target.value})}
            required className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500" />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Hours</label>
          <input type="number" min="0.5" max="24" step="0.5" value={form.hours}
            onChange={e => setForm({...form, hours: e.target.value})} required
            className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500" />
        </div>
      </div>
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-1">Description</label>
        <textarea value={form.description} onChange={e => setForm({...form, description: e.target.value})}
          rows={3} required className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 resize-none" />
      </div>
      <button type="submit" disabled={loading}
        className="w-full bg-blue-600 text-white py-2 rounded-lg text-sm font-medium hover:bg-blue-700 transition-colors disabled:opacity-60">
        {loading ? 'Logging...' : 'Log Hours'}
      </button>
    </form>
  )
}
