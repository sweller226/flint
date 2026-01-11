import { useRef } from 'react';
import Head from 'next/head';
import { gsap } from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';
import { ScrollToPlugin } from 'gsap/ScrollToPlugin';
import { useGSAP } from '@gsap/react';
import Image from 'next/image';
import logoPng from '../elements/logo.png';
import demoImage from '../elements/demo_image.png';

if (typeof window !== 'undefined') {
    gsap.registerPlugin(ScrollTrigger, ScrollToPlugin);
}

// --- Types ---
type CardProps = {
    label: string;
    description: string;
};

// --- Sub-Components ---
const FeatureBlock = ({ label, description }: { label: string; description: string }) => {
    return (
        <div className="relative w-full overflow-hidden rounded-[2rem] bg-[#121826] transition-all duration-700 hover:bg-[#151b2a] border border-white/[0.03] hover:border-white/10 shadow-2xl">

            <div className="absolute top-0 left-0 w-[2px] h-full bg-white/5 group-hover/block:bg-[#00FFA3] transition-all duration-700">
                {/* Small indicator dot at the top of the line */}
                <div className="absolute top-0 left-[-3px] w-2 h-2 rounded-full bg-white/10 group-hover/block:bg-[#00FFA3] group-hover/block:shadow-[0_0_10px_#00FFA3] transition-all duration-700" />
            </div>

            <div className="relative px-10 py-20 lg:px-16 lg:py-24 flex flex-col items-start">
                <h3 className="text-4xl lg:text-6xl font-[900] text-textMain uppercase tracking-[-0.06em] mb-6 leading-[0.9] max-w-3xl">
                    {label}
                </h3>

                <p className="text-lg lg:text-xl text-textMuted leading-relaxed font-medium max-w-xl opacity-60 group-hover/block:opacity-100 transition-opacity duration-700">
                    {description}
                </p>
            </div>
        </div>
    );
};

// --- Main Page ---
export default function Home() {
    const container = useRef<HTMLDivElement>(null);
    const terminalRef = useRef<HTMLDivElement>(null);
    const scrollRevealRef = useRef<HTMLDivElement>(null);
    const navBgRef = useRef<HTMLDivElement>(null);
    const documentationRef = useRef<HTMLDivElement>(null);
    const navRef = useRef<HTMLElement>(null);

    const scrollToBottom = () => {
        window.scrollTo({
            top: document.documentElement.scrollHeight,
            behavior: 'smooth'
        });
    };

    const scrollToSection = (elementRef: React.RefObject<HTMLDivElement>) => {
        if (elementRef.current) {
            gsap.to(window, {
                duration: 1.2,
                scrollTo: {
                    y: elementRef.current, // Ensure this is the .current property
                    offsetY: 80
                },
                ease: "power4.inOut"
            });
        }
    };

    useGSAP(() => {
        // 1. Hero Reveal
        const tl = gsap.timeline();
        tl.from(".hero-animate", { y: 40, opacity: 0, duration: 1, stagger: 0.15, ease: "power4.out" });

        // 2. Terminal Reveal Sequence
        gsap.timeline({
            scrollTrigger: {
                trigger: scrollRevealRef.current,
                start: "top 1%",
                end: "+=400",
                scrub: 1,
                pin: true,
                anticipatePin: 1
            }
        })
            .fromTo(terminalRef.current,
                { scale: 0.95, y: 40, opacity: 0, rotateX: 8 },
                { scale: 1, y: 0, opacity: 1, rotateX: 0, duration: 1.5 },
                0
            );

        const bgBar = navBgRef.current;

        if (bgBar) {
            ScrollTrigger.create({
                start: 'top top',
                end: 'max',
                onUpdate: (self) => {
                    if (self.direction === 1 && self.scroll() > 20) {
                        gsap.to(bgBar, { yPercent: -100, opacity: 0, duration: 0.2, ease: "power4.in" });
                    } else if (self.direction === -1) {
                        gsap.to(bgBar, { yPercent: 0, opacity: 1, duration: 0.2, ease: "power4.out" });
                    }
                }
            });
        }

        // Kinetic Axis Scroll Transitions
        const sections = gsap.utils.toArray('.feature-node');

        sections.forEach((node: any) => {
            const title = node.querySelector('.feature-title');
            const dot = node.querySelector('.feature-dot');
            const color = node.dataset.color;

            gsap.timeline({
                scrollTrigger: {
                    trigger: node,
                    // Starts the transition when the node is 95% from the top (right as it enters the screen)
                    start: "top 95%",
                    // Completes the transition when the node is 65% from the top (lower-third of the screen)
                    end: "top 65%",
                    scrub: 1,
                    toggleActions: "play reverse play reverse"
                }
            })
                .to(title, {
                    color: color,
                    opacity: 1,
                    duration: 1
                })
                .to(dot, {
                    backgroundColor: color,
                    boxShadow: `0 0 20px ${color}`,
                    scale: 1.25,
                    duration: 1
                }, 0);
        });

        const nav = navRef.current;

        if (nav) {
            ScrollTrigger.create({
                start: 0, // Start tracking immediately at the very top
                onUpdate: (self) => {
                    // Remove the "self.scroll() > 60" check for instant response
                    if (self.direction === 1) {
                        gsap.to(nav, {
                            yPercent: -100,
                            opacity: 0,
                            duration: 0.2, // Fast but fluid
                            ease: "none",  // Linear ease feels more responsive for "instant" actions
                            overwrite: "auto"
                        });
                    }
                    else if (self.direction === -1) {
                        gsap.to(nav, {
                            yPercent: 0,
                            opacity: 1,
                            duration: 0.2,
                            ease: "power2.out", // Keep a slight ease on return for polish
                            overwrite: "auto"
                        });
                    }
                }
            });
        }
    }, { scope: container });

    return (
        <main ref={container} className="min-h-screen bg-flint-bg text-flint-text-primary font-sans selection:bg-flint-primary selection:text-white">
            <Head>
                <title>Flint | ES Futures Terminal</title>
            </Head>

            {/* Navbar */}
            <nav ref={navRef} className="fixed top-0 left-0 w-full z-50 pointer-events-none">
                {/* Liquid Glass Background */}
                <div className="
                    absolute inset-0 h-20 lg:h-28 
                    bg-[#0B0F1A]/70 
                    backdrop-blur-xl 
                    backdrop-saturate-[180%] 
                    border-b border-white/[0.08] 
                    pointer-events-none
                " />

                {/* flex items-center handles the perfect vertical centering */}
                <div className="max-w-[1400px] mx-auto px-8 lg:px-12 h-20 lg:h-28 flex items-center justify-between pointer-events-auto relative z-20">

                    {/* Left Side: Logo */}
                    <div className="flex items-center cursor-pointer">
                        {/* Applied matching -translate-y-[2px] for perfect alignment with the button */}
                        <div className="relative w-20 h-20 lg:w-28 lg:h-28 flex items-center justify-center transform -translate-y-[5px]">
                            <Image
                                src={logoPng}
                                alt="Flint Logo"
                                priority
                                className="w-full h-full object-contain drop-shadow-[0_0_20px_rgba(77,185,231,0.3)]"
                            />
                        </div>
                    </div>

                    {/* Right Side: Download Button */}
                    <div className="hidden md:flex items-center">
                        <button
                            onClick={scrollToBottom}
                            className="
                                relative group overflow-hidden 
                                px-7 py-3 rounded-lg 
                                bg-[#4DB9E7] text-[#0B0F1A] 
                                /* Subtle 2px nudge for optical alignment with logo */
                                transform -translate-y-[2px] 
                                hover:scale-[1.02]
                                transition-all duration-300 ease-out 
                                shadow-[0_2px_10px_rgba(77,185,231,0.2)] 
                                hover:shadow-[0_4px_15px_rgba(77,185,231,0.3)]
                                pointer-events-auto
                                flex items-center justify-center
                            "
                        >
                            <div className="absolute inset-0 bg-white/10 opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
                            <span className="relative z-10 text-sm font-[900] uppercase tracking-[0.2em] whitespace-nowrap">
                                Download
                            </span>
                        </button>
                    </div>
                </div>
            </nav>

            <div className="pointer-events-none fixed inset-0 -z-10 bg-[radial-gradient(circle_at_top,_var(--primary),_transparent_55%),radial-gradient(circle_at_bottom,_var(--accent),_transparent_55%)] opacity-10" />

            {/* Hero Section */}
            <section className="px-6 pt-40 lg:pt-52 pb-20 relative">
                {/* Increased max-width to 1400px to give letters room to breathe */}
                <div className="max-w-[1400px] mx-auto text-center">

                    {/* - lg:text-[clamp(4rem,6vw,7.5rem)]: Scales dynamically with screen size
                        - tracking-[-0.02em]: Relaxed from -0.1em for better legibility 
                    */}
                    <h1 className="hero-animate text-5xl lg:text-[clamp(4rem,6vw,7.2rem)] font-[900] uppercase tracking-[-0.02em] mb-10 leading-[1.0] text-textMain">
                        Trade the <span className="text-primary">Momentum</span>. <br />
                        Master the <span className="text-[#00FFA3]">
                            <span className="relative inline-block">Spread</span>
                        </span>.
                    </h1>

                    <p className="hero-animate max-w-3xl mx-auto text-xl lg:text-2xl text-textMuted mb-16 italic font-medium leading-relaxed tracking-tight opacity-80">
                        &quot;Stop trading blind. Flint bridges the gap between retail Sim and Institutional Edge.&quot;
                    </p>

                    {/* Symmetric Buttons */}
                    <div className="hero-animate flex flex-col sm:flex-row justify-center gap-6 mb-24">
                        <button
                            onClick={scrollToBottom}
                            className="h-16 w-80 flex items-center justify-center rounded-2xl text-base font-[900] uppercase tracking-widest text-bgDark bg-primary hover:scale-105 transition-all group overflow-hidden relative shadow-2xl"
                        >
                            <span className="relative z-10">Download for Windows</span>
                        </button>

                        {/* ATTACHED HANDLER HERE */}
                        <button
                            onClick={() => scrollToSection(documentationRef)}
                            className="h-16 w-80 flex items-center justify-center rounded-2xl text-base font-[900] uppercase tracking-widest text-textMain border-2 border-borderDark bg-white/5 hover:bg-white/10 hover:scale-105 transition-all backdrop-blur-md"
                        >
                            Documentation
                        </button>
                    </div>
                </div>
            </section>

            {/* Terminal Reveal */}
            <section ref={scrollRevealRef} className="h-screen flex flex-col items-center justify-center overflow-hidden -mt-24">
                <div ref={terminalRef} className="w-full max-w-7xl px-8 lg:px-12 perspective-terminal">
                    <div className="relative rounded-[2.5rem] border-2 border-borderDark bg-panelDark shadow-[0_30px_100px_rgba(0,0,0,0.5)]">

                        {/* Ambient Outer Glow */}
                        <div className="absolute -inset-2 bg-gradient-to-r from-primary/20 to-accent/20 rounded-[2.7rem] blur-2xl opacity-20 animate-pulse" />

                        {/* Main Terminal Window with PNG Image */}
                        <div className="relative aspect-[21/10] bg-subpanelDark rounded-[2.3rem] overflow-hidden border border-white/5">

                            {/* The PNG Image */}
                            <Image
                                src={demoImage}
                                alt="Flint Terminal Demo"
                                fill
                                className="object-cover object-top opacity-90"
                                priority
                            />

                            {/* Glass/Reflective Overlay (Keep this for the "screen" effect) */}
                            <div className="absolute inset-0 bg-gradient-to-tr from-white/5 to-transparent pointer-events-none" />

                            {/* Subtle Scanline effect (Optional, adds to the terminal feel) */}
                            <div className="absolute inset-0 bg-[linear-gradient(rgba(18,16,16,0)_50%,rgba(0,0,0,0.25)_50%),linear-gradient(90deg,rgba(255,0,0,0.06),rgba(0,255,0,0.02),rgba(0,0,255,0.06))] bg-[length:100%_2px,3px_100%] pointer-events-none opacity-20" />
                        </div>
                    </div>
                </div>
            </section>

            {/* Kinetic Axis Section */}
            <section className="px-6 py-64 bg-bgDark relative overflow-hidden">
                <div className="absolute left-1/2 top-0 bottom-0 w-[1px] bg-gradient-to-b from-transparent via-white/20 to-transparent -translate-x-1/2 hidden lg:block" />

                <div className="max-w-[1400px] mx-auto relative flex flex-col gap-64">

                    {/* Node 1: Predictive Modeling - Muted Mint */}
                    <div className="feature-node flex flex-col lg:flex-row items-center justify-center" data-color="#94D2BD">
                        <div className="lg:w-[38%] text-right lg:pr-12">
                            <h3 className="feature-title text-5xl lg:text-[clamp(4rem,6vw,6.5rem)] font-[900] uppercase tracking-[-0.08em] leading-[0.82] text-white opacity-30 transition-all duration-700">
                                Predictive<br />Modeling
                            </h3>
                        </div>

                        <div className="relative w-12 h-12 hidden lg:flex items-center justify-center z-20">
                            <div className="feature-dot w-4 h-4 rounded-full bg-white/20 transition-all duration-500" />
                            <div className="absolute inset-0 border border-white/5 rounded-full scale-[2]" />
                        </div>

                        <div className="lg:w-[38%] lg:pl-12 mt-8 lg:mt-0">
                            <p className="text-xl lg:text-2xl text-textMuted font-medium max-w-sm leading-relaxed text-left opacity-80">
                                Machine learning-driven forecasted price targets based on ICT liquidity cycles.
                            </p>
                        </div>
                    </div>

                    {/* Node 2: Autonomous Execution - Muted Blue */}
                    <div className="feature-node flex flex-col lg:flex-row items-center justify-center" data-color="#4DB9E7">
                        <div className="lg:w-[38%] text-right lg:pr-12 order-2 lg:order-1">
                            <p className="text-xl lg:text-2xl text-textMuted font-medium max-w-sm ml-auto leading-relaxed opacity-80">
                                Advanced automated trading logic executing entries based on institutional bias and risk parameters.
                            </p>
                        </div>

                        <div className="relative w-12 h-12 hidden lg:flex items-center justify-center z-20 order-1 lg:order-2">
                            <div className="feature-dot w-4 h-4 rounded-full bg-white/20 transition-all duration-500" />
                            <div className="absolute inset-0 border border-white/5 rounded-full scale-[2]" />
                        </div>

                        <div className="lg:w-[38%] text-left lg:pl-12 order-3">
                            <h3 className="feature-title text-5xl lg:text-[clamp(4rem,6vw,6.5rem)] font-[900] uppercase tracking-[-0.08em] leading-[0.82] text-white opacity-30 transition-all duration-700">
                                Autonomous<br />Execution
                            </h3>
                        </div>
                    </div>

                    {/* Node 3: Solana Logging - Muted Purple */}
                    <div className="feature-node flex flex-col lg:flex-row items-center justify-center" data-color="#BB86FC">
                        <div className="lg:w-[38%] text-right lg:pr-12">
                            <h3 className="feature-title text-5xl lg:text-[clamp(4rem,6vw,6.5rem)] font-[900] uppercase tracking-[-0.08em] leading-[0.82] text-white opacity-30 transition-all duration-700">
                                Solana<br />Logging
                            </h3>
                        </div>
                        <div className="relative w-12 h-12 hidden lg:flex items-center justify-center z-20">
                            <div className="feature-dot w-4 h-4 rounded-full bg-white/20 transition-all duration-500" />
                            <div className="absolute inset-0 border border-white/5 rounded-full scale-[2]" />
                        </div>
                        <div className="lg:w-[38%] lg:pl-12 mt-8 lg:mt-0">
                            <p className="text-xl lg:text-2xl text-textMuted font-medium max-w-sm leading-relaxed text-left opacity-80">
                                Immutable trade logs on the Solana devnet for institutional-grade transparency.
                            </p>
                        </div>
                    </div>
                </div>
            </section>

            {/* Documentation Section */}
            <section ref={documentationRef} className="px-6 py-40 bg-bgDark relative border-t border-white/5">
                {/* Set to max-w-7xl (1280px) - the middle ground between 1400px and 1152px */}
                <div className="max-w-7xl mx-auto px-6 lg:px-10">

                    {/* Section Header */}
                    <div className="flex flex-col mb-20">
                        <h2 className="text-4xl lg:text-5xl font-[900] uppercase tracking-tight text-textMain mb-6">
                            Technical <span className="text-primary">Documentation</span>
                        </h2>
                        <div className="w-20 h-1 bg-primary" />
                    </div>

                    {/* Grid: gap-x-16 provides enough breathing room for the 1280px width */}
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-x-16 gap-y-16">

                        {/* Column 1: Architecture */}
                        <div className="flex flex-col">
                            <div className="mb-12 min-h-[140px]">
                                <h4 className="text-white font-bold uppercase tracking-[0.2em] text-sm mb-4 opacity-90">Core Architecture</h4>
                                <p className="text-textMuted text-base leading-relaxed opacity-80">
                                    Flint is an autonomous ES futures trading terminal built by a team blending institutional ML experience and software engineering.
                                </p>
                            </div>
                            <div>
                                <h4 className="text-white font-bold uppercase tracking-[0.2em] text-sm mb-4 opacity-90">The Toaster Model</h4>
                                <p className="text-textMuted text-base opacity-70 leading-relaxed">
                                    Our LSTM-based sequence model was trained on a decade of Databento CME data to forecast price dynamics at 1-minute resolution.
                                </p>
                            </div>
                        </div>

                        {/* Column 2: Execution */}
                        <div className="flex flex-col">
                            <div className="mb-12 min-h-[140px]">
                                <h4 className="text-white font-bold uppercase tracking-[0.2em] text-sm mb-4 opacity-90">Strategy & Execution</h4>
                                <p className="text-textMuted text-base leading-relaxed opacity-80">
                                    The system operates in Autonomous Mode for automated PnL and Signals Mode for discretionary trader confirmation.
                                </p>
                            </div>
                            <div>
                                <h4 className="text-white font-bold uppercase tracking-[0.2em] text-sm mb-4 opacity-90">Simulation Rigor</h4>
                                <p className="text-textMuted text-base opacity-70 leading-relaxed">
                                    We simulate live conditions by hiding future dataframes, forcing the agent to trade solely on historical context.
                                </p>
                            </div>
                        </div>

                        {/* Column 3: Trust */}
                        <div className="flex flex-col">
                            <div className="mb-12 min-h-[140px]">
                                <h4 className="text-white font-bold uppercase tracking-[0.2em] text-sm mb-4 opacity-90">Transparency Protocol</h4>
                                <p className="text-textMuted text-base leading-relaxed opacity-80">
                                    Leveraging Solana, Flint records intents on-chain, providing an immutable audit trail to prove signal validity.
                                </p>
                            </div>
                            <div>
                                <h4 className="text-white font-bold uppercase tracking-[0.2em] text-sm mb-4 opacity-90">Risk Parameters</h4>
                                <p className="text-textMuted text-base opacity-70 leading-relaxed">
                                    Reward functions are tuned to penalize inactivity while discouraging low-quality entries to find a genuine edge.
                                </p>
                            </div>
                        </div>

                    </div>
                </div>
            </section>

            {/* Final CTA */}
            <section className="py-32 px-6">
                <div className="max-w-5xl mx-auto rounded-[3rem] bg-gradient-to-br from-[#121826] to-[#0B0F1A] border border-white/5 py-24 text-center relative overflow-hidden shadow-2xl">
                    {/* The 'Phantom' Hue Spot */}
                    <div className="absolute -bottom-24 -left-24 w-64 h-64 bg-accent/10 blur-[80px]" />

                    <div className="relative z-10">
                        <h2 className="text-4xl font-bold mb-10 tracking-tight text-textMain">
                            Ready to trade with <span className="text-primary italic pr-1">Flint</span>?
                        </h2>

                        <button className="relative inline-flex items-center gap-6 px-12 py-6 bg-white text-bgDark font-black text-xl rounded-2xl hover:-translate-y-2 transition-all shadow-[0_20px_50px_rgba(255,255,255,0.1)]">
                            DOWNLOAD FOR WINDOWS
                            <div className="w-8 h-8 rounded-full bg-bgDark text-white flex items-center justify-center">
                                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M19 14l-7 7m0 0l-7-7m7 7V3" />
                                </svg>
                            </div>
                        </button>
                    </div>
                </div>
            </section>

            <footer className="py-12 text-center text-textMuted text-xs bg-bgDark">
                Copyright © 2026 Flint. All rights reserved.
            </footer>
        </main>
    );
}
