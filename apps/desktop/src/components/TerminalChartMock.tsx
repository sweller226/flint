import React from "react";
import { FlintChart } from "./FlintChart";
import { mockCandles } from "./mockCandles";

export const TerminalChartMock: React.FC = () => {
    return (
        <div className="h-full w-full rounded-2xl bg-[#050B10] p-1">
            <FlintChart candles={mockCandles} />
        </div>
    );
};
