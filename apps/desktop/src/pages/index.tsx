import React from "react";
import { TopBar } from "../components/TopBar";
import { ChartPanel } from "../components/ChartPanel";
import { RightRail } from "../components/RightRail";

export default function Home() {
    return (
        <div className="h-screen w-screen bg-black text-flint-text-primary flex overflow-hidden font-sans select-none">
            <div className="flex-1 flex flex-col min-w-0">
                <TopBar />

                <main className="flex-1 flex gap-4 px-6 pb-6 overflow-hidden min-h-0 bg-flint-bg">
                    <div className="flex-1 min-w-0">
                        <ChartPanel />
                    </div>

                    <aside className="w-[300px] flex flex-col gap-3 min-w-0">
                        <RightRail />
                    </aside>
                </main>
            </div>
        </div>
    );
}
