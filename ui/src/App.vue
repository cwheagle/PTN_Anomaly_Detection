<template>
  <div class="min-h-screen bg-slate-900 text-slate-100 p-6">
    <header class="mb-8 flex justify-between items-center border-b border-slate-700 pb-4">
      <div>
        <h1 class="text-2xl font-bold text-blue-400">PTN Anomaly Detection Dashboard</h1>
        <p class="text-slate-400 text-sm">Real-time Network Health Monitoring</p>
      </div>
      <div class="flex items-center gap-4">
        <!-- Scheduler Status -->
        <div class="flex items-center gap-2 bg-slate-800 px-4 py-2 rounded-lg border border-slate-700">
          <span class="text-xs text-slate-400 font-medium">Scheduler:</span>
          <span :class="['text-xs font-bold uppercase tracking-wider', 
                        schedulerStatus === 'running' ? 'text-emerald-400' : 'text-rose-400']">
            {{ schedulerStatus }}
          </span>
          <div class="flex gap-2 ml-2 border-l border-slate-700 pl-3">
            <button @click="controlScheduler('start')" 
                    v-if="schedulerStatus !== 'running'"
                    class="p-1 hover:bg-emerald-500/20 text-emerald-400 rounded transition-colors" title="Start Scheduler">
              <svg xmlns="http://www.w3.org/2000/svg" class="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>
            </button>
            <button @click="controlScheduler('stop')" 
                    v-if="schedulerStatus === 'running'"
                    class="p-1 hover:bg-rose-500/20 text-rose-400 rounded transition-colors" title="Stop Scheduler">
              <svg xmlns="http://www.w3.org/2000/svg" class="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="6" y="4" width="4" height="16"></rect><rect x="14" y="4" width="4" height="16"></rect></svg>
            </button>
          </div>
        </div>

        <div :class="['px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider', 
                      backendStatus === 'online' ? 'bg-emerald-500/20 text-emerald-400' : 'bg-rose-500/20 text-rose-400']">
          Backend: {{ backendStatus }}
        </div>
      </div>
    </header>

    <main class="grid grid-cols-1 lg:grid-cols-3 gap-6">
      <!-- Status Cards -->
      <div class="lg:col-span-3 grid grid-cols-1 md:grid-cols-3 gap-4 mb-2">
        <div class="bg-slate-800 p-4 rounded-xl border border-slate-700 flex justify-between items-center">
          <div>
            <p class="text-slate-400 text-xs uppercase font-bold tracking-wider mb-1">Next Run</p>
            <p class="text-xl font-mono text-blue-400">{{ nextRunTime || 'N/A' }}</p>
          </div>
          <div class="p-3 bg-blue-500/10 rounded-lg">
            <svg xmlns="http://www.w3.org/2000/svg" class="w-6 h-6 text-blue-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><polyline points="12 6 12 12 16 14"></polyline></svg>
          </div>
        </div>
        <div class="bg-slate-800 p-4 rounded-xl border border-slate-700 flex justify-between items-center">
          <div>
            <p class="text-slate-400 text-xs uppercase font-bold tracking-wider mb-1">Total Anomalies</p>
            <p class="text-xl font-mono text-emerald-400">{{ anomalies.length }}</p>
          </div>
          <div class="p-3 bg-emerald-500/10 rounded-lg">
            <svg xmlns="http://www.w3.org/2000/svg" class="w-6 h-6 text-emerald-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 12h-4l-3 9L9 3l-3 9H2"></path></svg>
          </div>
        </div>
        <div class="bg-slate-800 p-4 rounded-xl border border-slate-700 flex justify-between items-center">
          <div>
            <p class="text-slate-400 text-xs uppercase font-bold tracking-wider mb-1">Critical Alarms</p>
            <p class="text-xl font-mono text-rose-400">{{ alarms.length }}</p>
          </div>
          <div class="p-3 bg-rose-500/10 rounded-lg">
            <svg xmlns="http://www.w3.org/2000/svg" class="w-6 h-6 text-rose-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>
          </div>
        </div>
      </div>

      <!-- Alerts Panel -->
      <section class="lg:col-span-1 bg-slate-800 rounded-xl border border-slate-700 overflow-hidden flex flex-col h-[600px]">
        <div class="p-4 bg-slate-700/50 border-b border-slate-700 flex justify-between items-center">
          <h2 class="font-semibold flex items-center gap-2">
            <span class="w-2 h-2 bg-rose-500 rounded-full animate-pulse"></span>
            Real-time Critical Alarms
          </h2>
          <span class="text-xs bg-slate-600 px-2 py-0.5 rounded">{{ alarms.length }}</span>
        </div>
        <div class="flex-1 overflow-y-auto p-4 space-y-3">
          <div v-for="(alarm, idx) in alarms" :key="idx" 
               class="p-3 bg-rose-500/10 border border-rose-500/30 rounded-lg animate-in slide-in-from-right duration-300">
            <div class="flex justify-between items-start mb-1">
              <span class="font-mono text-xs text-rose-400">{{ alarm.event_time }}</span>
              <span class="text-[10px] bg-rose-500 text-white px-1.5 rounded uppercase font-bold">{{ alarm.severity }}</span>
            </div>
            <div class="text-sm font-semibold mb-1">{{ alarm.ip_addr }} (S{{ alarm.slot_id }}/P{{ alarm.port_id }})</div>
            <div class="text-xs text-slate-300">{{ alarm.message }}</div>
          </div>
          <div v-if="alarms.length === 0" class="text-center text-slate-500 py-10 text-sm italic">
            No critical alarms detected.
          </div>
        </div>
      </section>

      <!-- Dashboard Stats & Tables -->
      <section class="lg:col-span-2 space-y-6">
        <!-- Recent Anomalies -->
        <div class="bg-slate-800 p-6 rounded-xl border border-slate-700 shadow-xl">
          <div class="flex justify-between items-center mb-4">
            <h2 class="text-lg font-semibold flex items-center gap-2">
              <svg xmlns="http://www.w3.org/2000/svg" class="w-5 h-5 text-rose-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>
              Recent Anomalies
            </h2>
            <!-- Pagination Controls -->
            <div class="flex items-center gap-2" v-if="anomalies.length > 0">
              <button @click="anomaliesPage--" :disabled="anomaliesPage === 1" 
                      class="p-1.5 bg-slate-700 rounded hover:bg-slate-600 disabled:opacity-30 transition-colors">
                <svg xmlns="http://www.w3.org/2000/svg" class="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="15 18 9 12 15 6"></polyline></svg>
              </button>
              <span class="text-xs font-mono text-slate-400">{{ anomaliesPage }} / {{ Math.ceil(anomalies.length / pageSize) }}</span>
              <button @click="anomaliesPage++" :disabled="anomaliesPage >= Math.ceil(anomalies.length / pageSize)" 
                      class="p-1.5 bg-slate-700 rounded hover:bg-slate-600 disabled:opacity-30 transition-colors">
                <svg xmlns="http://www.w3.org/2000/svg" class="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="9 18 15 12 9 6"></polyline></svg>
              </button>
            </div>
          </div>
          <div class="overflow-x-auto min-h-[480px]">
            <table class="w-full text-left text-sm">
              <thead class="bg-slate-700/30 text-slate-400 uppercase text-[11px] tracking-wider">
                <tr>
                  <th class="px-4 py-3">Time</th>
                  <th class="px-4 py-3">IP Address</th>
                  <th class="px-4 py-3">Port</th>
                  <th class="px-4 py-3">Severity</th>
                  <th class="px-4 py-3">Reason</th>
                </tr>
              </thead>
              <tbody class="divide-y divide-slate-700">
                <tr v-for="item in pagedAnomalies" :key="item.occur_date + item.ip_addr + item.slot_id + item.port_id" class="hover:bg-slate-700/20 transition-colors">
                  <td class="px-4 py-3 font-mono text-xs">{{ item.occur_date }}</td>
                  <td class="px-4 py-3 font-semibold">{{ item.ip_addr }}</td>
                  <td class="px-4 py-3 text-slate-400">S{{ item.slot_id }}/P{{ item.port_id }}</td>
                  <td class="px-4 py-3">
                    <span :class="['px-2 py-0.5 rounded text-[10px] font-bold', 
                                  item.alarm_label === 'CRITICAL' ? 'bg-rose-500/20 text-rose-400 border border-rose-500/30' : 
                                  item.alarm_label === 'MAJOR' ? 'bg-orange-500/20 text-orange-400 border border-orange-500/30' : 
                                  'bg-blue-500/20 text-blue-400 border border-blue-500/30']">
                      {{ item.alarm_label }}
                    </span>
                  </td>
                  <td class="px-4 py-3 text-xs text-slate-400 truncate max-w-[200px]" :title="item.anomaly_reason">
                    {{ item.anomaly_reason }}
                  </td>
                </tr>
                <tr v-if="pagedAnomalies.length === 0">
                  <td colspan="5" class="px-4 py-10 text-center text-slate-500 italic">No anomalies detected. Everything looks stable.</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        <!-- Watchlist (Proactive Warnings) -->
        <div class="bg-slate-800 p-6 rounded-xl border border-slate-700 shadow-xl border-l-4 border-l-amber-500">
          <div class="flex justify-between items-center mb-4">
            <h2 class="text-lg font-semibold mb-4 flex justify-between items-center">
              <div class="flex items-center gap-2">
                <svg xmlns="http://www.w3.org/2000/svg" class="w-5 h-5 text-amber-500" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9"></path><path d="M13.73 21a2 2 0 0 1-3.46 0"></path></svg>
                Watchlist (Rising Trends)
              </div>
            </h2>
            <!-- Pagination Controls -->
            <div class="flex items-center gap-2" v-if="watchlist.length > 0">
              <button @click="watchlistPage--" :disabled="watchlistPage === 1" 
                      class="p-1.5 bg-slate-700 rounded hover:bg-slate-600 disabled:opacity-30 transition-colors">
                <svg xmlns="http://www.w3.org/2000/svg" class="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="15 18 9 12 15 6"></polyline></svg>
              </button>
              <span class="text-xs font-mono text-slate-400">{{ watchlistPage }} / {{ Math.ceil(watchlist.length / pageSize) }}</span>
              <button @click="watchlistPage++" :disabled="watchlistPage >= Math.ceil(watchlist.length / pageSize)" 
                      class="p-1.5 bg-slate-700 rounded hover:bg-slate-600 disabled:opacity-30 transition-colors">
                <svg xmlns="http://www.w3.org/2000/svg" class="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="9 18 15 12 9 6"></polyline></svg>
              </button>
            </div>
          </div>
          <div class="overflow-x-auto min-h-[480px]">
            <table class="w-full text-left text-sm">
              <thead class="bg-slate-700/30 text-slate-400 uppercase text-[11px] tracking-wider">
                <tr>
                  <th class="px-4 py-3">Time</th>
                  <th class="px-4 py-3">IP Address</th>
                  <th class="px-4 py-3">Port</th>
                  <th class="px-4 py-3">Severity</th>
                  <th class="px-4 py-3">Trend</th>
                  <th class="px-4 py-3">Estimated Time to Critical</th>
                </tr>
              </thead>
              <tbody class="divide-y divide-slate-700">
                <tr v-for="item in pagedWatchlist" :key="'watch-' + item.ip_addr + item.slot_id + item.port_id + item.occur_date" class="hover:bg-slate-700/20 transition-colors">
                  <td class="px-4 py-3 font-mono text-xs text-slate-400">{{ item.occur_date }}</td>
                  <td class="px-4 py-3 font-semibold">{{ item.ip_addr }}</td>
                  <td class="px-4 py-3 text-slate-400">S{{ item.slot_id }}/P{{ item.port_id }}</td>
                  <td class="px-4 py-3">
                    <div class="flex flex-col gap-1.5 min-w-[120px]">
                      <div class="flex justify-between items-center">
                        <span :class="['px-1.5 py-0.5 rounded text-[9px] font-bold uppercase', 
                                      item.alarm_label === 'MAJOR' ? 'bg-orange-500/20 text-orange-400 border border-orange-500/30' : 
                                      item.alarm_label === 'MINOR' ? 'bg-blue-500/20 text-blue-400 border border-blue-500/30' : 
                                      'bg-slate-500/20 text-slate-400 border border-slate-500/30']">
                          {{ item.alarm_label }}
                        </span>
                        <span class="text-[10px] font-mono text-slate-400">{{ item.severity.toFixed(1) }}%</span>
                      </div>
                      <div class="w-full bg-slate-700 rounded-full h-1">
                        <div :class="['h-1 rounded-full transition-all duration-500', 
                                     item.severity > 70 ? 'bg-orange-500' : 
                                     item.severity > 40 ? 'bg-blue-500' : 'bg-emerald-500']" 
                             :style="{ width: item.severity + '%' }"></div>
                      </div>
                    </div>
                  </td>
                  <td class="px-4 py-3">
                    <span class="flex items-center gap-1 text-amber-400 font-bold text-xs">
                      <svg xmlns="http://www.w3.org/2000/svg" class="w-3 h-3" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"></polyline><polyline points="17 6 23 6 23 12"></polyline></svg>
                      +{{ item.slope.toFixed(2) }}
                    </span>
                  </td>
                  <td class="px-4 py-3 font-mono text-xs">
                    <div v-if="item.ttf_minutes" class="flex flex-col gap-0.5">
                      <span class="text-blue-400 font-bold">{{ item.expected_fatal_time }}</span>
                      <span class="text-[10px] text-slate-500">({{ formatTTF(item.ttf_minutes) }} later)</span>
                    </div>
                    <span v-else class="text-slate-500 italic">Calculating...</span>
                  </td>
                </tr>
                <tr v-if="pagedWatchlist.length === 0">
                  <td colspan="6" class="px-4 py-10 text-center text-slate-500 italic">No rising trends detected.</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </section>
    </main>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted, computed } from 'vue'
import axios from 'axios'

const backendStatus = ref('offline')
const schedulerStatus = ref('unknown')
const nextRunTime = ref('')
const alarms = ref<any[]>([])
const anomalies = ref<any[]>([])
const watchlist = ref<any[]>([])
const sseSource = ref<EventSource | null>(null)

// Pagination State
const pageSize = 10
const anomaliesPage = ref(1)
const watchlistPage = ref(1)

const pagedAnomalies = computed(() => {
  const start = (anomaliesPage.value - 1) * pageSize
  return anomalies.value.slice(start, start + pageSize)
})

const pagedWatchlist = computed(() => {
  const start = (watchlistPage.value - 1) * pageSize
  return watchlist.value.slice(start, start + pageSize)
})

// TTF(분)를 "HHh MMm" 형식으로 변환
const formatTTF = (minutes: number | null) => {
  if (minutes === null || minutes === undefined) return 'N/A'
  const h = Math.floor(minutes / 60)
  const m = Math.round(minutes % 60)
  return h > 0 ? `${h}h ${m}m` : `${m}m`
}

const fetchAnomalies = async () => {
  try {
    // 1등급(MINOR) 이상인 모든 이상치 조회
    const res = await axios.get('/api/anomalies?severity_min=1')
    anomalies.value = res.data
    backendStatus.value = 'online'
    
    const maxPage = Math.ceil(anomalies.value.length / pageSize) || 1
    if (anomaliesPage.value > maxPage) anomaliesPage.value = maxPage
  } catch (err) {
    console.error('Failed to fetch anomalies', err)
    backendStatus.value = 'offline'
  }
}

const fetchWatchlist = async () => {
  try {
    // 3등급(CRITICAL) 미만이면서 상승 추세인 데이터 조회
    const res = await axios.get('/api/anomalies?severity_max=2&rising_only=true')
    watchlist.value = res.data
    
    const maxPage = Math.ceil(watchlist.value.length / pageSize) || 1
    if (watchlistPage.value > maxPage) watchlistPage.value = maxPage
  } catch (err) {
    console.error('Failed to fetch trend data', err)
  }
}

const fetchSchedulerStatus = async () => {
  try {
    const res = await axios.get('/api/scheduler/status')
    schedulerStatus.value = res.data.status
    nextRunTime.value = res.data.next_run_time
  } catch (err) {
    console.error('Failed to fetch scheduler status', err)
  }
}

const controlScheduler = async (action: string) => {
  try {
    if (action === 'start') schedulerStatus.value = 'running'
    else if (action === 'stop') schedulerStatus.value = 'stopped'
    
    const res = await axios.post('/api/scheduler/status', { action })
    schedulerStatus.value = res.data.status
    nextRunTime.value = res.data.next_run_time
  } catch (err) {
    console.error('Failed to control scheduler', err)
    alert('Failed to control scheduler')
    fetchSchedulerStatus()
  }
}

const setupSSE = () => {
  if (sseSource.value) sseSource.value.close()
  
  const source = new EventSource('/api/stream/alarms')
  sseSource.value = source
  
  source.addEventListener('alarm', (event) => {
    const data = JSON.parse(event.data)
    alarms.value.unshift(data)
    if (alarms.value.length > 20) alarms.value.pop()
    fetchAnomalies()
    fetchWatchlist()
  })

  source.onerror = (err) => {
    console.error('SSE error:', err)
    source.close()
    setTimeout(setupSSE, 5000)
  }
}

onMounted(() => {
  fetchAnomalies()
  fetchWatchlist()
  fetchSchedulerStatus()
  setupSSE()
  
  const interval = setInterval(() => {
    fetchAnomalies()
    fetchWatchlist()
    fetchSchedulerStatus()
  }, 60000)
  onUnmounted(() => clearInterval(interval))
})

onUnmounted(() => {
  if (sseSource.value) sseSource.value.close()
})
</script>

<style>
@keyframes slide-in {
  from { transform: translateX(100%); opacity: 0; }
  to { transform: translateX(0); opacity: 1; }
}
.animate-in {
  animation: slide-in 0.3s ease-out forwards;
}
</style>
