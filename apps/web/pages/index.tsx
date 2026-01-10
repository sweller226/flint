import Head from 'next/head'

export default function Home() {
    return (
        <main className="min-h-screen bg-flint-bg text-flint-text-primary font-sans selection:bg-flint-primary selection:text-white">
            <Head>
                <title>Flint | ES Futures Terminal</title>
            </Head>

            {/* Gradient glow */}
            <div className="pointer-events-none fixed inset-0 -z-10 bg-[radial-gradient(circle_at_top,_rgba(37,99,235,0.1),_transparent_55%),radial-gradient(circle_at_bottom,_rgba(168,85,247,0.08),_transparent_55%)]" />

            {/* Hero */}
            <section className="px-6 pt-16 pb-20 lg:px-20 lg:pt-24 lg:pb-24">
                <div className="max-w-5xl mx-auto text-center">
                    {/* Logo / wordmark */}
                    <div className="inline-flex items-center gap-2 rounded-full border border-flint-border bg-white px-3 py-1 text-xs text-flint-text-secondary mb-6 shadow-sm">
                        <span className="h-1.5 w-1.5 rounded-full bg-flint-secondary animate-pulse" />
                        <span className="font-medium">Flint · ES Futures Terminal</span>
                    </div>

                    <h1 className="text-4xl sm:text-5xl lg:text-[3.5rem] font-bold tracking-tight mb-4 text-flint-text-primary">
                        Start Trading <span className="text-flint-primary">Smarter</span>.
                    </h1>

                    <p className="max-w-2xl mx-auto text-sm sm:text-base text-flint-text-secondary mb-8 leading-relaxed">
                        A minimal, high‑frequency terminal for ES day traders. Built on <span className="text-flint-primary font-medium">ICT concepts</span>,
                        order‑flow volume, and <span className="text-flint-accent font-medium">generative AI</span> insights.
                    </p>

                    {/* CTAs */}
                    <div className="flex flex-col sm:flex-row justify-center gap-4 mb-12">
                        <button
                            className="inline-flex items-center justify-center rounded-xl px-6 py-3 text-sm font-semibold text-white
                         bg-flint-primary shadow-lg shadow-blue-500/20 hover:bg-blue-700 transition-all hover:-translate-y-0.5"
                        >
                            Download for Windows
                        </button>
                        <button
                            className="inline-flex items-center justify-center rounded-xl px-6 py-3 text-sm font-semibold
                         text-flint-text-primary border border-flint-border bg-white hover:bg-slate-50 transition-all shadow-sm"
                        >
                            View Documentation
                        </button>
                    </div>
                </div>

                {/* Screenshot frame */}
                <div className="max-w-5xl mx-auto mt-6">
                    <div className="rounded-2xl border border-flint-border bg-white shadow-2xl shadow-slate-200/50 overflow-hidden group relative p-1">
                        <div className="rounded-xl border border-flint-border overflow-hidden bg-flint-bg aspect-[16/10] flex items-center justify-center relative">
                            {/* Abstract UI representation */}
                            <div className="text-flint-text-secondary text-sm font-medium">
                                [App Screenshot Placeholder]
                            </div>
                        </div>
                    </div>
                </div>
            </section>

            {/* Features row */}
            <section className="bg-white border-t border-flint-border">
                <div className="max-w-5xl mx-auto px-6 py-20 lg:px-20 grid gap-6 sm:grid-cols-3 text-sm">
                    <FeatureCard
                        label="ICT Native"
                        icon="🪙"
                        description="FVG detection, liquidity pools, session highs and Market Structure Shifts out of the box."
                    />
                    <FeatureCard
                        label="Gemini AI"
                        icon="✨"
                        description="Ask strategy questions in plain English and get bias, risk, and suggested parameters."
                    />
                    <FeatureCard
                        label="Solana Logging"
                        icon="🔗"
                        description="Write every simulated trade to Solana devnet for an immutable, shareable audit trail."
                    />
                </div>
            </section>

            <footer className="py-8 text-center text-flint-text-secondary text-xs">
                built for the 2026 Solana AI Hackathon
            </footer>
        </main>
    );
}

type CardProps = { label: string; icon: string; description: string };

const FeatureCard = ({ label, icon, description }: CardProps) => (
    <div className="rounded-2xl border border-flint-border bg-flint-bg/50 px-5 py-6 hover:border-flint-primary/50 transition-colors group cursor-default">
        <div className="inline-flex items-center gap-3 mb-3">
            <span className="text-xl">{icon}</span>
            <span className="text-sm font-bold tracking-wide text-flint-text-primary group-hover:text-flint-primary transition-colors">
                {label}
            </span>
        </div>
        <p className="text-xs text-flint-text-secondary leading-relaxed">{description}</p>
    </div>
);
