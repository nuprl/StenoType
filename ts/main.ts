import * as ts from "typescript";

// Code adapted from:
// https://github.com/GammaTauAI/opentau/blob/main/ts-compiler/main.ts
// https://github.com/GammaTauAI/opentau/blob/main/evaluator/scripts/ts-does-parse/main.ts

const compilerOptions = {
    target: ts.ScriptTarget.Latest,
    module: ts.ModuleKind.CommonJS,
    strict: false,
    noImplicitAny: false,
    noImplicitThis: false,
    noEmit: true,
    noImplicitReturns: false,
    allowJs: true,
    checkJs: true,
};

const defaultCompilerHost = ts.createCompilerHost(compilerOptions);

const makeCompilerHost = (
    filename: string,
    sourceFile: ts.SourceFile
): ts.CompilerHost => ({
    getSourceFile: (name, languageVersion) => {
        if (name === filename) {
            return sourceFile;
        } else {
            return defaultCompilerHost.getSourceFile(name, languageVersion);
        }
    },
    writeFile: (_filename, _data) => {},
    getDefaultLibFileName: () =>
        defaultCompilerHost.getDefaultLibFileName(compilerOptions),
    useCaseSensitiveFileNames: () => false,
    getCanonicalFileName: (filename) => filename,
    getCurrentDirectory: () => "",
    getNewLine: () => "\n",
    getDirectories: () => [],
    fileExists: () => true,
    readFile: () => "",
});

const createProgram = (code: string, setParentNodes = false): ts.Program => {
    const prog = ts.createProgram({
        rootNames: ["file.ts"],
        options: compilerOptions,
        host: makeCompilerHost(
            "file.ts",
            ts.createSourceFile(
                "file.ts",
                code,
                ts.ScriptTarget.Latest,
                setParentNodes,
                ts.ScriptKind.TS
            )
        ),
    });
    return prog;
};

let buffer = "";
process.stdin.on("data", (chunk) => {
    buffer = buffer.concat(chunk.toString());
});

process.stdin.on("close", () => {
    const program = createProgram(buffer);
    const file = program.getSourceFile("file.ts")!;
    const typecheckDiag = ts.getPreEmitDiagnostics(program, file);
    const syntaxDiag = program.getSyntacticDiagnostics(file);

    const result = JSON.stringify({
        type_errors: typecheckDiag.length,
        parse_errors: syntaxDiag.length,
    });
    console.log(result);
});
