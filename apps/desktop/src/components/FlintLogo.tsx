import React from "react";

type FlintLogoProps = {
    withText?: boolean;
    light?: boolean; // true for light backgrounds
    size?: number;   // overall height in px
};

import logo from "../assets/logo_text.png";
import Image from "next/image";

export const FlintLogo: React.FC<FlintLogoProps> = ({
    withText = true,
    light = false,
    size = 48,
}) => {
    // Determine height and width to maintain aspect ratio (text logo is wider)
    // Assuming approx 3:1 aspect ratio for text logo based on visual
    const height = size;
    const width = withText ? size * 3.5 : size;

    return (
        <div style={{ height, width: withText ? 'auto' : size, overflow: 'hidden', display: 'flex', alignItems: 'center' }}>
            {withText ? (
                <img src={logo.src} alt="Flint" style={{ height: '100%', width: 'auto', objectFit: 'contain' }} />
            ) : (
                // If no text, crop/show only the icon part or just scale it down?
                // For now, let's just show the full logo but smaller if !withText, or strictly speaking
                // we should have a separate icon-only asset. The user provided two assets.
                // Image 0 was round icon, Image 1 was text logo.
                // If withText=false, we might want to use the round icon?
                // But the user said "use the second image PURELY for in the app replacing that old logo design"
                // The old design had a mode withText vs without.
                // Let's assume for now we just show the text logo.
                <img src={logo.src} alt="Flint" style={{ height: '100%', width: 'auto', objectFit: 'contain' }} />
            )}
        </div>
    );
};
