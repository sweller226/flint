import '@/index.css'
import type { AppProps } from 'next/app'
import { SimulationProvider } from '../context/SimulationContext'

export default function App({ Component, pageProps }: AppProps) {
    return (
        <SimulationProvider>
            <Component {...pageProps} />
        </SimulationProvider>
    )
}
