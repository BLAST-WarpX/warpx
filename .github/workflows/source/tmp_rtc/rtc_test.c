/* TEMPORARY: self-test for run_without_crt_dialogs.py, compiled with /MDd /RTC1 */
__declspec(dllexport) int rtc_uninit (int flag)
{
    int x;
    if (flag) { x = 1; }
    return x; /* /RTCu: used without being initialized if flag == 0 */
}
