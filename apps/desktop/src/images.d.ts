declare module '*.png' {
    const content: import('next/dist/client/image').StaticImageData;
    export default content;
}

declare module '*.svg' {
    const content: any;
    export default content;
}

declare module '*.jpg' {
    const content: import('next/dist/client/image').StaticImageData;
    export default content;
}

declare module '*.jpeg' {
    const content: import('next/dist/client/image').StaticImageData;
    export default content;
}

declare module '*.gif' {
    const content: import('next/dist/client/image').StaticImageData;
    export default content;
}

declare module '*.ico' {
    const content: any;
    export default content;
}
