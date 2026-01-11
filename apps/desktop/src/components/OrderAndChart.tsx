import React from "react";
import { OrderPanel } from "./OrderPanel";
import { ChartPanel } from "./ChartPanel";

export const OrderAndChart = () => (
    <div className="flex flex-1 border-r border-flint-border bg-flint-panel overflow-hidden">
        {/* Order side */}
        <div className="w-[340px] border-r border-flint-border flex flex-col">
            <OrderPanel />
        </div>

        {/* Chart */}
        <div className="flex-1 flex flex-col overflow-hidden">
            <ChartPanel />
        </div>
    </div>
);
