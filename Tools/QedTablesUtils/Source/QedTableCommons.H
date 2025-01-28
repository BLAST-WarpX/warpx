#ifndef QED_TABLES_COMMONS_
#define QED_TABLES_COMMONS_

#include <iostream>
#include <string>

template <typename ContainerType, typename ElementType>
bool Contains (const ContainerType& container, const ElementType& el)
{
    return container.find(el) != std::end(container);
}

void AbortWithMessage(const std::string& msg)
{
    std::cout << "### ABORT : " << msg << "\n";
    std::cout << "___________________________\n";
    std::cout.flush();
    exit(1);
}

void SuccessExit()
{
    std::cout << "___________________________\n";
    exit(0);
}

#endif //QED_TABLES_COMMONS_
