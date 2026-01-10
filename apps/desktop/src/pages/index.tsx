import React from "react";
import { Sidebar } from "../components/Sidebar";
import { TopBar } from "../components/TopBar";
import { OrderPanel } from "../components/OrderPanel";
import { ChartPanel } from "../components/ChartPanel";
import { RightRail } from "../components/RightRail";

export default function Home() {
    return (
        <div className="h-screen w-screen bg-flint-bg text-flint-text-primary flex overflow-hidden font-sans select-none">
            <Sidebar />

            <div className="flex-1 flex flex-col min-w-0">
                <TopBar />

                <main className="flex-1 flex gap-4 px-4 pb-4 overflow-hidden min-h-0">
                    {/* Left: order + chart */}
                    <section className="flex-1 flex gap-4 min-w-0">
                        <div className="w-[320px] rounded-2xl bg-flint-panel shadow-sm border border-flint-border p-4 flex flex-col">
                            <OrderPanel />
                        </div>
                        <div className="flex-1 rounded-2xl bg-flint-panel shadow-sm border border-flint-border p-3 flex flex-col min-w-0">
                            <ChartPanel />
                        </div>
                    </section>

                    {/* Right: signals + AI */}
                    <aside className="w-[320px] flex flex-col gap-4 min-w-0">
                        <RightRail />
                    </aside>
                </main>
            </div>
        </div>
    );
}
