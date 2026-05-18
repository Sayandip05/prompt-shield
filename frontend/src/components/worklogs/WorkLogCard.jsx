export default function WorkLogCard({ log }) {
  if (!log) return null
  return (
    <div className="bg-white border border-gray-100 rounded-xl p-4 flex items-center justify-between">
      <div>
        <p className="text-sm font-medium text-gray-900">{log.description || 'Work logged'}</p>
        <p className="text-xs text-gray-500 mt-0.5">{log.date}</p>
      </div>
      <span className="text-sm font-semibold text-blue-600">{log.hours}h</span>
    </div>
  )
}
