import React from "react";

type FlintLogoProps = {
    withText?: boolean;
    light?: boolean; // true for light backgrounds
    size?: number;   // overall height in px
};

export const FlintLogo: React.FC<FlintLogoProps> = ({
    withText = true,
    light = false,
    size = 48,
}) => {
    const textFill = light ? "#111827" : "#F9FAFB";

    return (
        <svg
            width={withText ? size * 4 : size}
            height={size}
            viewBox={withText ? "0 0 200 60" : "0 0 48 48"}
            xmlns="http://www.w3.org/2000/svg"
        >
            <defs>
                <linearGradient id="flintGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" stopColor="#00FFA3" />
                    <stop offset="45%" stopColor="#03E1FF" />
                    <stop offset="100%" stopColor="#DC1FFF" />
                </linearGradient>
            </defs>

            {/* mark */}
            <g transform={withText ? "translate(4,4)" : ""}>
                <rect x="0" y="0" width="48" height="48" rx="16" fill="url(#flintGradient)" />
                <rect x="16" y="12" width="8" height="24" rx="4" fill="#020814" />
                <rect x="16" y="12" width="14" height="8" rx="4" fill="#020814" />
                <path d="M34 14 L40 10 L40 18 Z" fill="#020814" opacity="0.9" />
            </g>

            {withText && (
                <g transform="translate(64, 15)">
                    <text
                        x="0"
                        y="22"
                        fontFamily="Inter, system-ui, -apple-system, BlinkMacSystemFont, sans-serif"
                        fontSize="24"
                        fontWeight="600"
                        letterSpacing="1.2"
                        fill={textFill}
                    >
                        FLINT
                    </text>
                </g>
            )}
        </svg>
    );
};
