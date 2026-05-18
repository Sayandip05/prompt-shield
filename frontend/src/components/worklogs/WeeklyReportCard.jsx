export default function WeeklyReportCard({ report }) {
  if (!report) return null
  return (
    <div className="bg-white border border-gray-100 rounded-xl p-4">
      <p className="text-sm font-medium text-gray-700">Week of {report.week_start}</p>
      <p className="text-gray-600 mt-1">{report.summary || 'No summary provided.'}</p>
    </div>
  )
}
