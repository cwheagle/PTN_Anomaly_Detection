import { reactive, ref } from 'vue'
import axios from 'axios'

export const store = reactive({
  backendStatus: 'offline',
  schedulerStatus: 'unknown',
  nextRunTime: '',
  alarms: [] as any[],
  anomalies: [] as any[],
  watchlist: [] as any[],
  modelStatus: {} as Record<string, any>,
  isRefreshingAnomalies: false,
  isRefreshingWatchlist: false,
  isRefreshingModelStatus: false,
  _refreshTimer: null as any,

  debouncedFetch(delay = 1000) {
    if (this._refreshTimer) clearTimeout(this._refreshTimer)
    this._refreshTimer = setTimeout(() => {
      this.fetchAnomalies()
      this.fetchWatchlist()
      this._refreshTimer = null
      console.log('[Store] Dashboard data refreshed (globally debounced)')
    }, delay)
  },

  async fetchActiveAlarms() {
    try {
      // 최신 시점의 Critical 알람만 가져오기
      const res = await axios.get('/api/anomalies?severity_min=3')
      this.alarms = res.data.map((a: any) => ({
        type: 'ALARM',
        event_time: a.occur_date,
        ip_addr: a.ip_addr,
        slot_id: a.slot_id,
        port_id: a.port_id,
        message: a.anomaly_reason
      }))
      console.log('[Store] Active alarms synchronized from DB:', this.alarms.length)
    } catch (err) {
      console.error('Failed to sync active alarms', err)
    }
  },

  async fetchAnomalies() {
    this.isRefreshingAnomalies = true
    try {
      const res = await axios.get('/api/anomalies?severity_min=1')
      this.anomalies = res.data
      this.backendStatus = 'online'
    } catch (err) {
      console.error('Failed to fetch anomalies', err)
      this.backendStatus = 'offline'
    } finally {
      this.isRefreshingAnomalies = false
    }
  },

  async fetchWatchlist() {
    this.isRefreshingWatchlist = true
    try {
      const res = await axios.get('/api/anomalies?severity_max=2&rising_only=true')
      this.watchlist = res.data
    } catch (err) {
      console.error('Failed to fetch trend data', err)
    } finally {
      this.isRefreshingWatchlist = false
    }
  },

  async fetchSchedulerStatus() {
    try {
      const res = await axios.get('/api/scheduler/status')
      this.schedulerStatus = res.data.status
      this.nextRunTime = res.data.next_run_time
    } catch (err) {
      console.error('Failed to fetch scheduler status', err)
    }
  },

  async controlScheduler(action: string) {
    try {
      if (action === 'start') this.schedulerStatus = 'running'
      else if (action === 'stop') {
        this.schedulerStatus = 'stopped'
        this.alarms = []
      }
      
      const res = await axios.post('/api/scheduler/status', { action })
      this.schedulerStatus = res.data.status
      this.nextRunTime = res.data.next_run_time
      
      if (this.schedulerStatus !== 'running') {
        this.alarms = []
      }
    } catch (err) {
      console.error('Failed to control scheduler', err)
      this.fetchSchedulerStatus()
      throw err
    }
  },

  async runNow() {
    try {
      const res = await axios.post('/api/scheduler/run-now')
      return res.data
    } catch (err) {
      console.error('Failed to trigger manual run', err)
      throw err
    }
  },

  async fetchHistory(params: { ip_addr: string, slot_id: number, port_id: number, days?: number }) {
    try {
      const res = await axios.get(`/api/anomalies/history`, {
        params: { ...params, days: params.days || 1 }
      })
      return res.data
    } catch (err) {
      console.error('Failed to fetch history in store', err)
      throw err
    }
  },

  async fetchModelStatus() {
    this.isRefreshingModelStatus = true
    try {
      const res = await axios.get('/api/model/status')
      this.modelStatus = res.data
    } catch (err) {
      console.error('Failed to fetch model status', err)
    } finally {
      this.isRefreshingModelStatus = false
    }
  },

  async updateInferenceConfig(ft: string, settings: any) {
    try {
      const res = await axios.post(`/api/model/inference-config?ft=${ft}`, settings)
      await this.fetchModelStatus()
      return res.data
    } catch (err) {
      console.error('Failed to update inference config', err)
      throw err
    }
  },

  async trainModel(ft: string, training_config: any = {}, date_params: any = {}) {
    try {
      const res = await axios.post(`/api/model/train`, training_config, {
        params: { 
          ft,
          train_start: date_params.train_start,
          train_end: date_params.train_end,
          test_start: date_params.test_start,
          test_end: date_params.test_end
        }
      })
      return res.data
    } catch (err) {
      console.error('Failed to trigger training', err)
      throw err
    }
  },

  async stopTraining(ft: string) {
    try {
      const res = await axios.post(`/api/model/train/stop?ft=${ft}`)
      await this.fetchModelStatus()
      return res.data
    } catch (err) {
      console.error('Failed to stop training', err)
      throw err
    }
  }
})
