/* Copyright 2022 Luca Fedeli
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef ABLASTR_TEXT_MSG_H_
#define ABLASTR_TEXT_MSG_H_

#include <string>

namespace ablastr::utils::TextMsg
{
    /** \brief This function formats a text message as an error message,
    *   adding the '### ERROR: ' prefix and (by default) performing text wrapping
    *
    * @param[in] msg the string to be formatted
    * @param[in] do_text_wrapping if true, the text of the message is automatically wrapped
    *
    * @return the formatted message
    */
    std::string
    Err (const std::string &msg, bool do_text_wrapping = true);

    /** \brief This function formats a text message as an info message,
    *   adding the '### INFO: ' prefix and (by default) performing text wrapping
    *
    * @param[in] msg the string to be formatted
    * @param[in] do_text_wrapping if true, the text of the message is automatically wrapped
    *
    * @return the formatted message
    */
    std::string
    Info (const std::string &msg, bool do_text_wrapping = true);

    /** \brief This function formats a text message as a warning message,
    *   adding the '### WARN: ' prefix and (by default) performing text wrapping.
    *   Warning: this format is not used by the WarningLogger, which has an internal,
    *   dedicated, formatter.
    *
    * @param[in] msg the string to be formatted
    * @param[in] do_text_wrapping if true, the text of the message is automatically wrapped
    *
    * @return the formatted message
    */
    std::string
    Warn (const std::string &msg, bool do_text_wrapping = true);

    /** \brief This function is a wrapper around amrex::Assert, designed to ensure the uniform
    * formatting of the error messages. The function is designed to be used via the
    * ABLASTR_ALWAYS_ASSERT_WITH_MESSAGE(EX,MSG) macro.
    *
    * @param[in] ex the text of the failed assertion
    * @param[in] file the file where the assertion failed
    * @param[in] line the line where the assertion failed
    * @param[in] msg the error message
    */
    void
    Assert (const char *ex, const char *file, int line, const std::string &msg);

    /** \brief This function is a wrapper around amrex::Abort, designed to ensure the uniform
    * formatting of the error messages. The function is designed to be used via the
    * ABLASTR_ABORT_WITH_MESSAGE(MSG) macro.
    *
    * @param[in] file the file where abort was called
    * @param[in] line the line here abort was called
    * @param[in] msg the error message
    */
    void
    Abort (const char *file, int line, const std::string &msg);

} // namespace ablastr::utils::TextMsg

#define ABLASTR_ALWAYS_ASSERT_WITH_MESSAGE(EX,MSG) (EX)?((void)0) : ablastr::utils::TextMsg::Assert( # EX , __FILE__, __LINE__ , MSG)

#define ABLASTR_ABORT_WITH_MESSAGE(MSG) ablastr::utils::TextMsg::Abort( __FILE__, __LINE__ , MSG)

#endif // ABLASTR_TEXT_MSG_H_
