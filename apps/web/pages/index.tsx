import Head from 'next/head'

export default function Home() {
    return (
        <main className="min-h-screen bg-[#020814] text-slate-100 font-sans selection:bg-[#00FFA3] selection:text-[#001219]">
            <Head>
                <title>Flint | ES Futures Terminal</title>
            </Head>

            {/* Gradient glow */}
            <div className="pointer-events-none fixed inset-0 -z-10 bg-[radial-gradient(circle_at_top,_rgba(56,189,248,0.22),_transparent_55%),radial-gradient(circle_at_bottom,_rgba(236,72,153,0.16),_transparent_55%)]" />

            {/* Hero */}
            <section className="px-6 pt-16 pb-20 lg:px-20 lg:pt-24 lg:pb-24">
                <div className="max-w-5xl mx-auto text-center">
                    {/* Logo / wordmark */}
                    <div className="inline-flex items-center gap-2 rounded-full border border-slate-800 bg-black/30 px-3 py-1 text-xs text-slate-400 mb-6 backdrop-blur">
                        <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 animate-pulse" />
                        <span className="font-mono">Flint · ES Futures Terminal</span>
                    </div>

                    <h1 className="text-4xl sm:text-5xl lg:text-[3.5rem] font-bold tracking-tight mb-4">
                        <span className="bg-gradient-to-r from-[#00FFA3] via-[#03E1FF] to-[#DC1FFF] bg-clip-text text-transparent">
                            FLINT
                        </span>
                    </h1>

                    <p className="max-w-2xl mx-auto text-sm sm:text-base text-slate-300 mb-8 leading-relaxed">
                        A minimal, high‑frequency terminal for ES day traders. Built on <span className="text-[#00FFA3]">ICT concepts</span>,
                        order‑flow volume, and <span className="text-[#03E1FF]">generative AI</span> insights.
                    </p>

                    {/* CTAs */}
                    <div className="flex flex-col sm:flex-row justify-center gap-4 mb-12">
                        <button
                            className="inline-flex items-center justify-center rounded-md px-6 py-3 text-sm font-bold text-[#001219]
                         bg-gradient-to-r from-[#00FFA3] via-[#03E1FF] to-[#DC1FFF] shadow-[0_0_25px_rgba(56,189,248,0.45)] hover:opacity-95 hover:scale-105 transition-all"
                        >
                            Download for Windows
                        </button>
                        <button
                            className="inline-flex items-center justify-center rounded-md px-6 py-3 text-sm font-bold
                         text-slate-100 border border-slate-700 bg-black/40 hover:bg-slate-900/60 transition-all backdrop-blur"
                        >
                            View Documentation
                        </button>
                    </div>
                </div>

                {/* Screenshot frame */}
                <div className="max-w-5xl mx-auto mt-6">
                    <div className="rounded-2xl border border-slate-800/80 bg-[#040C18] shadow-[0_18px_60px_rgba(0,0,0,0.8)] overflow-hidden group relative">
                        <div className="border-b border-slate-800 px-4 py-2 flex items-center justify-between text-[11px] text-slate-500 bg-[#020612]">
                            <div className="flex gap-1.5">
                                <div className="w-2.5 h-2.5 rounded-full bg-red-500/20" />
                                <div className="w-2.5 h-2.5 rounded-full bg-yellow-500/20" />
                                <div className="w-2.5 h-2.5 rounded-full bg-green-500/20" />
                            </div>
                            <div className="font-mono opacity-50">Flint desktop · ES Futures · Sim mode</div>
                            <div className="w-8" />
                        </div>

                        <div className="aspect-[16/10] bg-[#001219] flex items-center justify-center relative overflow-hidden">
                            {/* Abstract UI representation if no screenshot */}
                            <div className="absolute inset-0 flex">
                                <div className="w-16 border-r border-[#10141C] bg-[#050B10]"></div>
                                <div className="flex-1 flex flex-col">
                                    <div className="h-10 border-b border-[#10141C] bg-[#050B10]"></div>
                                    <div className="flex-1 flex">
                                        <div className="w-1/4 border-r border-[#10141C] bg-[#050B10]"></div>
                                        <div className="flex-1 bg-gradient-to-tr from-[#001219] to-[#011c26] flex items-center justify-center">
                                            <span className="text-slate-700 tracking-[0.2em] font-light text-xs uppercase animate-pulse">Terminal Active</span>
                                        </div>
                                    </div>
                                </div>
                                <div className="w-64 border-l border-[#10141C] bg-[#050B10]"></div>
                            </div>
                        </div>
                    </div>
                </div>
            </section>

            {/* Features row */}
            <section className="bg-[#020612] border-t border-slate-900/70">
                <div className="max-w-5xl mx-auto px-6 py-20 lg:px-20 grid gap-6 sm:grid-cols-3 text-sm">
                    <FeatureCard
                        label="ICT Native"
                        accent="from-emerald-400/80 to-emerald-500"
                        description="FVG detection, liquidity pools, session highs and Market Structure Shifts out of the box."
                    />
                    <FeatureCard
                        label="Gemini AI"
                        accent="from-sky-400/80 to-sky-500"
                        description="Ask strategy questions in plain English and get bias, risk, and suggested parameters."
                    />
                    <FeatureCard
                        label="Solana Logging"
                        accent="from-fuchsia-400/80 to-fuchsia-500"
                        description="Write every simulated trade to Solana devnet for an immutable, shareable audit trail."
                    />
                </div>
            </section>

            <footer className="py-8 text-center text-slate-800 text-xs text-slate-600">
                built for the 2026 Solana AI Hackathon
            </footer>
        </main>
    );
}

type CardProps = { label: string; accent: string; description: string };

const FeatureCard = ({ label, accent, description }: CardProps) => (
    <div className="rounded-xl border border-slate-800 bg-black/30 px-5 py-6 hover:border-slate-700 transition-colors group cursor-default">
        <div className="inline-flex items-center gap-3 mb-3">
            <span className={`h-4 w-4 rounded-full bg-gradient-to-br ${accent} group-hover:scale-110 transition-transform`} />
            <span className="text-sm font-bold tracking-wide text-slate-200 group-hover:text-white transition-colors">
                {label}
            </span>
        </div>
        <p className="text-xs text-slate-400 leading-relaxed font-light">{description}</p>
    </div>
);
