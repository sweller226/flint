import React from "react";
import { Sidebar } from "../components/Sidebar";
import { TopBar } from "../components/TopBar";
import { OrderPanel } from "../components/OrderPanel";
import { ChartPanel } from "../components/ChartPanel";
import { RightRail } from "../components/RightRail";

export default function Home() {
    return (
        <div className="h-screen w-screen bg-black text-flint-text-primary flex overflow-hidden font-sans select-none">
            <Sidebar />

            <div className="flex-1 flex flex-col min-w-0">
                <TopBar />

                <main className="flex-1 flex gap-4 px-6 pb-6 overflow-hidden min-h-0 bg-flint-bg">
                    {/* Left: order + chart */}
                    <section className="flex-1 flex gap-3 min-w-0">
                        <div className="w-[300px] rounded-xl bg-flint-panel border border-flint-border flex flex-col overflow-hidden">
                            <OrderPanel />
                        </div>
                        <div className="flex-1 rounded-xl bg-flint-panel border border-flint-border flex flex-col min-w-0 overflow-hidden">
                            <ChartPanel />
                        </div>
                    </section>

                    {/* Right: signals + AI */}
                    <aside className="w-[300px] flex flex-col gap-3 min-w-0">
                        <RightRail />
                    </aside>
                </main>
            </div>
        </div>
    );
}
