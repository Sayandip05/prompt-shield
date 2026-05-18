/**
 * Format an ISO date string or Date object to a human-readable string.
 * @param {string|Date} date
 * @param {object} options  Intl.DateTimeFormat options
 * @returns {string}
 */
export function formatDate(date, options = { year: 'numeric', month: 'short', day: 'numeric' }) {
  if (!date) return '—'
  try {
    return new Intl.DateTimeFormat('en-US', options).format(new Date(date))
  } catch {
    return String(date)
  }
}

export default formatDate
