#include <windows.h>
#include <iostream>
#include <string>
#include <vector>
#include <locale>       // For std::locale
#include <codecvt>


using namespace std;


//function to convert std::string (UTF-8) to std::wstring (UTF-16)
//This conversion is crucial for Windows API functions when compiled for Unicode.
std::wstring s2ws(const std::string& str) {
    try {
        using convert_type = std::codecvt_utf8_utf16<wchar_t>;
        std::wstring_convert<convert_type, wchar_t> converter;
        return converter.from_bytes(str);
    } catch (const std::range_error& e) {
        // Handle conversion errors (e.g., invalid UTF-8 sequence)
        std::wcerr << L"Error converting string to wstring: " << e.what() << std::endl;
        return std::wstring(); // Return empty or handle as appropriate
    }
}



int executeWindowsProcess(const string &path, const string &args){

    //convert the path to wstring (UTF-16) for Windows API compatibility
    wstring wPath = s2ws(path);
    wstring wArgs = s2ws(args);

    //1st element is the path of theb executable
    //if it contains some spaces, it must be enclosed in quotes to escape them
    wstring fullCmd_w;
    if (wPath.find(L' ') != std::wstring::npos) { // Note L' ' for wide char space
        fullCmd_w = L"\"" + wPath + L"\""; // Note L"" for wide string literals
    } else {
        fullCmd_w = wPath;
    }

    // argv1,2.... are the params
    if (!wArgs.empty()) {
        fullCmd_w += L" " + wArgs;
    }

    //convert string to LPSTR (LPSTR is a pointer to a null-terminated string)
    //use a buffer since CreateProcess expects a writable string
    vector<wchar_t> fullCmdBuffer(fullCmd_w.begin(), fullCmd_w.end());
    //null-terminate the string
    fullCmdBuffer.push_back(L'\0');

    //create STARTUPINFO, PROCESS_INFORMATION structures for Windows API
    STARTUPINFOW si;
    PROCESS_INFORMATION pi;

    //initialize the STARTUPINFO structure
    ZeroMemory(&si, sizeof(si));
    //set the size of the structure
    si.cb = sizeof(si);
    //initialize the PROCESS_INFORMATION structure
    ZeroMemory(&pi, sizeof(pi));

    // Extract the directory from wPath to set as current directory for the child process
    std::wstring workingDir_w = wPath;
    size_t lastSlashPos = workingDir_w.find_last_of(L"\\/");
    if (lastSlashPos != std::wstring::npos) {
        workingDir_w = workingDir_w.substr(0, lastSlashPos);
    } else {
        workingDir_w = L"."; // If no path separator, assume current directory
    }


    //create and start the process
    //current directory is set to NULL, so it gets set to the directory of the executable
    if (!CreateProcessW(      //CreateProcessW -> explicitly use the wide-character version
        NULL,                 // lpApplicationName: NULL meaning we pass the command line directly
        fullCmdBuffer.data(), // lpCommandLine
        NULL,                 // lpProcessAttributes
        NULL,                 // lpThreadAttributes
        FALSE,                // bInheritHandles: don't inherit handles
        0,                    // dwCreationFlags: creation flags (0 for default)
        NULL,                 // lpEnvironment: use the parent's environment
        workingDir_w.c_str(), // lpCurrentDirectory: SET WORKING DIRECTORY NOT OF ORCHESTRATOR BUT OF THE EXECUTABLE
        &si,                  // lpStartupInfo: Information about how to start the process
        &pi                   // lpProcessInformation: Information about the created process
    )) {
        std::wcerr << L"CreateProcessW failed (" << GetLastError() << L").\n";
        return 1;
    } 


    std::wcout << L"Started Process with PID: " << pi.dwProcessId << L" fullCmd: " << fullCmd_w << std::endl;
    //then, wait for completion of the started process
    WaitForSingleObject(pi.hProcess, INFINITE);

    //catch the exit code of the process
    DWORD exitCode;
    if (GetExitCodeProcess(pi.hProcess, &exitCode)) {
        wcout << L"Process " << pi.dwProcessId << L" exited with code: " << exitCode << endl;
    } else {
        wcerr << L"Process " << pi.dwProcessId << L" failed to get exit code (" << GetLastError() << L").\n";
    }

    //close process and thread handles
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);

    return 0;
}

int main(void){
    const string seq_node_it_v1_path = "../algorithms/sequential_node_it/main_v1.exe";
    const string seq_node_it_v2_path = "../algorithms/sequential_node_it/main_v2.exe";

    //execution
    executeWindowsProcess(seq_node_it_v1_path, "graph_100.g RTX_4060_R");

    cout << "Sequential Node IT v1 executed successfully." << endl;

    return 0;
}